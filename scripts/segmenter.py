import pandas as pd
import csv
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cc3d
import numpy as np
import psutil
import shutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from monai.apps.auto3dseg.transforms import EnsureSameShaped
from monai.auto3dseg.utils import datafold_read
from monai.bundle.config_parser import ConfigParser
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, list_data_collate, TestTimeAugmentation
from monai.inferers import SlidingWindowInferer
from monai.losses import DeepSupervisionLoss, DiceCELoss, FocalLoss, TverskyLoss
from monai.metrics import CumulativeAverage, do_metric_reduction, compute_dice
from monai.metrics.utils import ignore_background
from monai.networks.utils import one_hot
from model import get_network
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.transforms import (
    AsDiscreted,
    Compose,
    ConcatItemsd,
    CopyItemsd,
    CropForegroundd,
    DataStatsd,
    DeleteItemsd,
    EnsureTyped,
    Invertd,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    Rand3DElasticd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ResampleToMatchd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)
from monai.utils import MetricReduction, convert_to_dst_type, optional_import, set_determinism
from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from typing import Dict, Hashable, Mapping, List, Optional

suv_threshold = 0.2 

class LabelEmbedClassIndex(MapTransform):
    """
    Label embedding according to class_index
    """
    def __init__(
        self,
        keys: KeysCollection = "label",
        allow_missing_keys: bool = False,
        class_index: Optional[List] = None,

    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be compared to the source_key item shape.
            allow_missing_keys: do not raise exception if key is missing.
            class_index: a list of class indices
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.class_index = class_index

    def label_mapping(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return torch.cat([sum([x==i for i in c]) for c in self.class_index], dim=0).to(dtype=dtype)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        if self.class_index is not None:
            for key in self.key_iterator(d):
                d[key] = self.label_mapping(d[key])
        return d

class DiceHelper:
    def __init__(
        self,
        sigmoid: bool = False,
        include_background: Optional[bool] = None,
        to_onehot_y: Optional[bool] = None,
        softmax: Optional[bool] = None,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN_BATCH,
        get_not_nans: bool = True,
        ignore_empty: bool = True,
        activate: bool = False,
    ) -> None:
        super().__init__()

        self.sigmoid = sigmoid

        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty

        self.include_background = sigmoid if include_background is None else include_background
        if self.include_background:
            raise ValueError("include_background must be False when sigmoid is False")
        self.to_onehot_y = not sigmoid if to_onehot_y is None else to_onehot_y
        self.softmax = not sigmoid if softmax is None else softmax
        self.activate = activate

    def __call__(self, y_pred: Union[torch.Tensor, list], y: torch.Tensor):

        n_pred_ch = y_pred.shape[1]
      
        if self.softmax:
            if n_pred_ch > 1:
                y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
                y_pred = one_hot(y_pred, num_classes=n_pred_ch, dim=1)
        elif self.sigmoid:
            if self.activate:
                y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float()

        if self.to_onehot_y and n_pred_ch > 1 and y.shape[1] == 1:
            y = one_hot(y, num_classes=n_pred_ch, dim=1)

        data = compute_dice(
            y_pred=y_pred, y=y, include_background=self.include_background, ignore_empty=self.ignore_empty
        )
        f, not_nans = do_metric_reduction(data, self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 26 # 18 or 26
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp

def false_pos_pix(gt_array, pred_array, pred_array_baseline=None):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    
    false_pos = 0 
    false_pos_num = 0
    for idx in range(1, min(pred_conn_comp.max()+1, 50)):
        comp_mask = np.isin(pred_conn_comp, idx)
        if comp_mask.sum() <= 8: # ignore small connected components 
            continue
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum() 
            false_pos_num = false_pos_num+1
            
    return false_pos_num

def false_neg_pix(gt_array, pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    
    false_neg = 0
    true_pos = 0
    false_neg_num = 0 
    true_pos_num = 0
    for idx in range(1, min(gt_conn_comp.max()+1, 50)):
        comp_mask = np.isin(gt_conn_comp, idx) 
        if (comp_mask*pred_array).sum() == 0:
            false_neg = false_neg+comp_mask.sum()
            false_neg_num = false_neg_num+1
        else:
            true_pos = true_pos+comp_mask.sum()
            true_pos_num = true_pos_num+1
            
    return true_pos_num, false_neg_num 
    
class TPFPFNHelper:
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, y_pred, y):
        n_pred_ch = y_pred.shape[1]
        if n_pred_ch > 1:
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
        else:
            raise ValueError("y_pred must have more than 1 channel, use softmax instead")
        
        n_gt_ch = y.shape[1]
        if n_gt_ch > 1: 
            y = torch.argmax(y, dim=1, keepdim=True)
        
        # reducing only spatial dimensions (not batch nor channels)
        TP_sum = 0 
        FP_sum = 0
        FN_sum = 0
        y_copy = deepcopy(y).detach().cpu().numpy().squeeze()
        y_pred_copy = deepcopy(y_pred).detach().cpu().numpy().squeeze()
        if y_copy.ndim == 3: # if batch dim is reduced 
            y_copy = y_copy[np.newaxis, ...]
            y_pred_copy = y_pred_copy[np.newaxis, ...]

        for ii in range(y_copy.shape[0]):
            y_ = y_copy[ii]
            y_pred_ = y_pred_copy[ii]   

            FP = false_pos_pix(y_, y_pred_)
            TP, FN = false_neg_pix(y_, y_pred_)

            TP_sum += TP
            FP_sum += FP
            FN_sum += FN

        return TP_sum, FP_sum, FN_sum # all are volumes

def logits2pred(logits, sigmoid=False, dim=1):
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return torch.softmax(logits, dim=dim) if not sigmoid else torch.sigmoid(logits)

class DataTransformBuilder:
    def __init__(
        self,
        roi_size: list,
        image_key: str = "image",
        label_key: str = "label",
        resample: bool = False,
        resample_resolution: Optional[list] = None,
        normalize_mode: str = "PET",
        normalize_params: Optional[dict] = None,
        crop_mode: str = "ratio",
        crop_params: Optional[dict] = None,
        extra_modalities: Optional[dict] = None,
        custom_transforms=None,
        debug: bool = False,
        rank: int = 0,
        **kwargs,
    ) -> None:

        self.roi_size, self.image_key, self.label_key = roi_size, image_key, label_key

        self.resample, self.resample_resolution = resample, resample_resolution
        self.normalize_mode = normalize_mode
        self.normalize_params = normalize_params if normalize_params is not None else {}
        self.crop_mode = crop_mode
        self.crop_params = crop_params if crop_params is not None else {}

        self.extra_modalities = extra_modalities if extra_modalities is not None else {}
        self.custom_transforms = custom_transforms if custom_transforms is not None else {}

        self.extra_options = kwargs
        self.debug = debug
        self.rank = rank

    def get_custom(self, key):
        return self.custom_transforms.get(key, [])

    def get_load_transforms(self):

        ts = self.get_custom("load_transforms")
        if len(ts) > 0:
            return ts

        keys = [self.image_key, self.label_key] + list(self.extra_modalities)
        ts.append(LoadImaged(keys=keys, ensure_channel_first=True, dtype=None, allow_missing_keys=True, image_only=True))
        ts.append(EnsureTyped(keys=keys, data_type="tensor", dtype=torch.float, allow_missing_keys=True))
        ts.append(EnsureSameShaped(keys=self.label_key, source_key=self.image_key, allow_missing_keys=True))
        ts.extend(self.get_custom("after_load_transforms"))

        return ts

    def threshold_for_pet(self, x):
        # threshold at 0.2
        return x > suv_threshold # SUV uptake = 1.0 +- 10% is a normal SUV uptake 

    def get_resample_transforms(self, resample_label=True, crop_foreground=True):

        ts = self.get_custom("resample_transforms")
        if len(ts) > 0:
            return ts

        keys = [self.image_key] 
        if resample_label:
            keys += [self.label_key]

        extra_keys = self.extra_modalities # dict
        # self.image_key is PET, self.extra_modalities is CT
        mode = ["bilinear", "nearest"] if resample_label else ["bilinear"]
        
        if crop_foreground:
            ts.append(CropForegroundd(keys=keys, source_key=self.image_key, select_fn=self.threshold_for_pet, margin=0, allow_missing_keys=True, allow_smaller=True)) # it can be accomplished in a pre-processing step 

        if self.resample:
            if self.resample_resolution is None:
                raise ValueError("resample_resolution is not provided")
            pixdim = self.resample_resolution
            ts.append(
                Spacingd(
                    keys=keys,
                    pixdim=pixdim,
                    mode=mode,
                    dtype=torch.float,
                    allow_missing_keys=True,
                )
            )
        
        # match extra modalities to the key image. 
        for extra_key in extra_keys:
            ts.append(ResampleToMatchd(keys=extra_key, key_dst=self.image_key, dtype=np.float32))
        
        ts.extend(self.get_custom("after_resample_transforms"))

        return ts

    def get_normalize_transforms(self):

        ts = self.get_custom("normalize_transforms")
        if len(ts) > 0:
            return ts

        modalities = {self.image_key: self.normalize_mode} # default input is PET_interim
        modalities.update(self.extra_modalities)

        for key, normalize_mode in modalities.items():
            normalize_mode = normalize_mode.lower()
            if "pet" in normalize_mode: # SUV input 
                intensity_bounds = [0, 30] # 0-30, 0-10
                ts.append(ScaleIntensityRanged(keys=key, a_min=intensity_bounds[0], a_max=intensity_bounds[1], b_min=0, b_max=1, clip=True))                
            elif "ct" in normalize_mode:
                intensity_bounds = [-150, 250] 
                ts.append(ScaleIntensityRanged(keys=key, a_min=intensity_bounds[0], a_max=intensity_bounds[1], b_min=-1, b_max=1, clip=False))
                ts.append(Lambdad(keys=key, func=lambda x: torch.sigmoid(x))) # scale to 0-1
            else:
                raise ValueError("Unsupported normalize_mode" + str(self.normalize_mode))
            
        if len(modalities) > 0: 
            ts.append(ConcatItemsd(keys=list(modalities), name=self.image_key))  # concatenate all modalities at the channels 
            ts.append(DeleteItemsd(keys=list(self.extra_modalities)))  # release memory
        else:
            raise ValueError("No enough modalities")

        ts.extend(self.get_custom("after_normalize_transforms"))
        return ts

    def get_crop_transforms(self):

        ts = self.get_custom("crop_transforms")
        if len(ts) > 0:
            return ts

        if self.roi_size is None:
            raise ValueError("roi_size is not specified")

        keys = [self.image_key, self.label_key]
        ts = []
        ts.append(SpatialPadd(keys=keys, spatial_size=self.roi_size))
        if self.crop_mode == "ratio":
            output_classes = self.crop_params.get("output_classes", None)
            if output_classes is None:
                raise ValueError("crop_params option output_classes must be specified")

            crop_ratios = self.crop_params.get("crop_ratios", None)
            num_samples = self.crop_params.get("num_samples", 1) 

            ts.append(
                RandCropByLabelClassesd(
                    keys=keys,
                    label_key=self.label_key,
                    num_classes=len(crop_ratios), # default = output_classes (0: bg, 1: follow-up only, 2: baseline-only, 3: baseline+follow-up)
                    spatial_size=self.roi_size,
                    num_samples=num_samples, 
                    ratios=crop_ratios, # [1, 2, 2, 4]
                    warn = False,
                )
            )
        elif self.crop_mode == "rand":
            ts.append(RandSpatialCropd(keys=keys, roi_size=self.roi_size, random_size=False))
        else:
            raise ValueError("Unsupported crop mode" + str(self.crop_mode))

        ts.extend(self.get_custom("after_crop_transforms"))

        return ts
    
    def get_augment_transforms(self):

        ts = self.get_custom("augment_transforms")
        if len(ts) > 0:
            return ts

        if self.roi_size is None:
            raise ValueError("roi_size is not specified")

        ts = []
        ts.append(
            RandAffined(
                keys=[self.image_key, self.label_key],
                prob=0.2,
                rotate_range=[0.25, 0.25, 0.25],
                scale_range=[0.2, 0.2, 0.2],
                mode=["bilinear", "nearest"],
                # spatial_size=self.roi_size,
                cache_grid=True,
                padding_mode="border",
            )
        )
        ts.append(
            Rand3DElasticd(
                keys=[self.image_key, self.label_key],
                prob=0.2,
                sigma_range=(3,7),
                magnitude_range=(10,30),
                rotate_range=[0.25, 0.25, 0.25],
                scale_range=[0.2, 0.2, 0.2],
                mode=["bilinear", "nearest"],
                spatial_size=self.roi_size,
                cache_grid=True,
                padding_mode="border",
            )
        )
        ts.append(
            RandGaussianSmoothd(
                keys=self.image_key, prob=0.2, sigma_x=[0.5, 1.0], sigma_y=[0.5, 1.0], sigma_z=[0.5, 1.0]
            )
        )
        ts.append(RandGaussianNoised(keys=self.image_key, prob=0.2, mean=0.0, std=0.1))

        ts.append(RandFlipd(keys=[self.image_key, self.label_key], prob=0.5, spatial_axis=0))
        ts.append(RandFlipd(keys=[self.image_key, self.label_key], prob=0.5, spatial_axis=1))
        ts.append(RandFlipd(keys=[self.image_key, self.label_key], prob=0.5, spatial_axis=2))

        ts.extend(self.get_custom("after_augment_transforms"))

        return ts

    def get_final_transforms(self):
        return self.get_custom("final_transforms")

    @classmethod
    def get_postprocess_transform(
        cls, save_mask=False, invert=False, transform=None, sigmoid=False, save_prob=False, output_path=None, output_postfix=""
    ) -> Compose:

        ts = []
        if invert and transform is not None:
            ts.append(Invertd(keys="pred", orig_keys="image", transform=transform, nearest_interp=False))

        if save_mask and output_path is not None:
            ts.append(CopyItemsd(keys="pred", times=1, names="seg"))
            if not save_prob:
                ts.append(AsDiscreted(keys="seg", argmax=True) if not sigmoid else AsDiscreted(keys="seg", threshold=0.5))
            ts.append(
                SaveImaged(
                    keys=["seg"],
                    output_dir=output_path,
                    output_postfix=output_postfix,
                    output_dtype=np.float32,
                    separate_folder=False,
                    squeeze_end_dims=True,
                    resample=False,
                )
            )

        return Compose(ts)

    def __call__(self, augment=False, resample_label=False) -> Compose:

        ts = []
        ts.extend(self.get_load_transforms())
        ts.extend(self.get_resample_transforms(resample_label=resample_label))
        ts.extend(self.get_normalize_transforms())

        if augment:
            ts.extend(self.get_crop_transforms())
            ts.extend(self.get_augment_transforms())

        ts.extend(self.get_final_transforms())

        return Compose(ts)

    def __repr__(self) -> str:

        out: str = f"DataTransformBuilder: with image_key: {self.image_key}, label_key: {self.label_key} \n"
        out += f"roi_size {self.roi_size} resample {self.resample} resample_resolution {self.resample_resolution} \n"
        out += f"normalize_mode {self.normalize_mode} normalize_params {self.normalize_params} \n"
        out += f"crop_mode {self.crop_mode} crop_params {self.crop_params} \n"
        out += f"extra_modalities {self.extra_modalities} \n"
        for k, trs in self.custom_transforms.items():
            out += f"Custom {k} : {str(trs)} \n"
        return out


class Segmenter:
    def __init__(
        self, config_file: Optional[Union[str, Sequence[str]]] = None, config_dict: Dict = {}, rank: int = 0
    ) -> None:

        self.rank = rank
        self.distributed = dist.is_initialized()

        if rank == 0:
            print("Segmenter", rank, config_file, config_dict)

        np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
        warnings.filterwarnings(
            action="ignore", module=r"monai\.transforms\.utils", lineno=564
        )  # silence warning about missing class in groundtruth

        if "fork" in mp.get_all_start_methods():
            mp.set_start_method("fork", force=True)  # lambda functions fail to pickle without it
        else:
            warnings.warn("Multiprocessing method fork is not available, some non-picklable objects (e.g. lambda ) may fail")

        parser, config = self.parse_input_config(config_file=config_file, override=config_dict, rank=rank)

        if config["cuda"] and torch.cuda.is_available():
            self.device = torch.device(self.rank)
            if self.distributed and dist.get_backend() == dist.Backend.NCCL:
                torch.cuda.set_device(rank)
        else:
            self.device = torch.device("cpu")

        if rank == 0:
            print(yaml.safe_dump(config))
        self.config = config
        self.parser = parser

        if config["ckpt_path"] is not None and not os.path.exists(config["ckpt_path"]):
            os.makedirs(config["ckpt_path"], exist_ok=True)

        if config["determ"]:
            set_determinism(seed=716)
        elif torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        parser.config["network"]["resolution"] = config["resample_resolution"]
        parser.parse(reset=True)
        img_size = config["roi_size"] # 96 or 128 
        in_channels = config["input_channels"]
        n_class = config["output_classes"]
        use_checkpoint = config["use_checkpoint"]
        model = get_network(img_size, in_channels, n_class, use_checkpoint)

        if config["pretrained_ckpt_name"] is not None:
            self.checkpoint_load(ckpt=config["pretrained_ckpt_name"], model=model)

        model = model.to(self.device)

        if self.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(
                module=model, device_ids=[rank], output_device=rank, find_unused_parameters=False
            )

        if rank == 0:
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Total parameters count", pytorch_total_params, "distributed", self.distributed)

        self.model = model
        self.loss_function = parser.get_parsed_content("loss")
        self.loss_function = DeepSupervisionLoss(self.loss_function)

        self.acc_function = [TPFPFNHelper(), DiceHelper(sigmoid=config["sigmoid"])]
        self.grad_scaler = GradScaler(enabled=config["amp"])

        if parser.get("sliding_inferrer") is not None:
            self.sliding_inferrer = parser.get_parsed_content("sliding_inferrer")
        else:
            self.sliding_inferrer = SlidingWindowInferer(
                roi_size=config["roi_size"],
                sw_batch_size=config["sw_batch_size"],
                overlap=0.625,
                mode="gaussian",
                cache_roi_weight_map=True,
                progress=True,
                cpu_thresh=512**3 // config["output_classes"],
            )

        # check for custom transforms
        custom_transforms = {}
        for tr in config.get("custom_data_transforms", []):

            must_include_keys = ("key", "path", "transform")
            if not all(k in tr for k in must_include_keys):
                raise ValueError("custom transform must include " + str(must_include_keys))

            if os.path.abspath(tr["path"]) not in sys.path:
                sys.path.append(os.path.abspath(tr["path"]))

            custom_transforms.setdefault(tr["key"], [])
            custom_transforms[tr["key"]].append(ConfigParser(tr["transform"]).get_parsed_content())

        if len(custom_transforms) > 0 and rank == 0:
            print("Using custom transforms", custom_transforms)


        if isinstance(config["class_index"], list) and len(config["class_index"])>0:
            # custom label embedding, if class_index provided
            custom_transforms.setdefault("final_transforms", [])
            custom_transforms["final_transforms"].append(LabelEmbedClassIndex(keys="label", class_index=config["class_index"], allow_missing_keys=True))

        self.data_tranform_builder = DataTransformBuilder(
            roi_size=config["roi_size"],
            resample=config["resample"],
            resample_resolution=config["resample_resolution"],
            normalize_mode=config["normalize_mode"],
            normalize_params={"intensity_bounds": config["intensity_bounds"]},
            crop_mode=config["crop_mode"],
            crop_params={"output_classes": config["output_classes"], "crop_ratios": \
                                config["crop_ratios"], "num_samples": config["num_samples"]},
            extra_modalities=config["extra_modalities"],
            custom_transforms=custom_transforms,
        )

        self.lr_scheduler = None
        self.optimizer = None


    def parse_input_config(
        self, config_file: Optional[Union[str, Sequence[str]]] = None, override: Dict = {}, rank: int = 0
    ) -> Tuple[ConfigParser, Dict]:

        config = ConfigParser.load_config_files(config_file)

        config.setdefault("finetune", {"enabled": False, "ckpt_name": None})
        config.setdefault("validate", {"enabled": False, "ckpt_name": None, "save_mask": True, "output_path": None})
        config.setdefault("infer", {"enabled": False, "ckpt_name": None})

        parser = ConfigParser(config=config)
        parser.update(pairs=override)

        if config.get("data_file_base_dir", None) is None or config.get("data_list_file_path", None) is None:
            raise ValueError("CONFIG: data_file_base_dir and  data_list_file_path must be provided")

        if config.get("bundle_root", None) is None:
            config["bundle_root"] = str(Path(__file__).parent.parent)

        if "sigmoid" not in config:
            config["sigmoid"] = not config.get("softmax", True)

        if "modality" not in config:
            if rank == 0:
                warnings.warn("CONFIG: modality is not provided, assuming MRI")
            config["modality"] = "PET"

        if "normalize_mode" not in config:
            config["normalize_mode"] = "range" if config["modality"].lower() == "ct" else "meanstd"
            if rank == 0:
                print("CONFIG: normalize_mode is not provided, assuming: ", config["normalize_mode"])

        # assign defaults
        config.setdefault("loss", None)
        config.setdefault("acc", None)
        config.setdefault("amp", True)
        config.setdefault("cuda", True)
        config.setdefault("fold", 0)
        config.setdefault("batch_size", 1)
        config.setdefault("num_epochs", 300)
        config.setdefault("num_warmup_epochs", 5)
        config.setdefault("num_epochs_per_validation", 2)
        config.setdefault("num_epochs_per_saving", 2)
        config.setdefault("determ", True)
        config.setdefault("quick", True)
        config.setdefault("cache_rate", None)
        config.setdefault("ckpt_path", None)
        config.setdefault("ckpt_save", True)
        config.setdefault("crop_mode", "ratio")
        config.setdefault("crop_ratios", None)
        config.setdefault("resample_resolution", [1.0, 1.0, 1.0])
        config.setdefault("resample", False)
        config.setdefault("roi_size", [1024, 1024, 1024])
        config.setdefault("num_workers", 4)
        config.setdefault("extra_modalities", {})
        config.setdefault("intensity_bounds", [-250, 250])

        config.setdefault("class_index", None)
        config.setdefault("class_names", [])
        if not isinstance(config["class_names"], (list, tuple)):
            config["class_names"] = []

        pretrained_ckpt_name = None
        if config["validate"]["enabled"]:
            pretrained_ckpt_name = config["validate"]["ckpt_name"]
        elif config["infer"]["enabled"]:
            pretrained_ckpt_name = config["infer"]["ckpt_name"]
        elif config["finetune"]["enabled"]:
            pretrained_ckpt_name = config["finetune"]["ckpt_name"]
        config["pretrained_ckpt_name"] = pretrained_ckpt_name

        if not torch.cuda.is_available() and config["cuda"]:
            print("No cuda is available.! Running on CPU!!!")
            config["cuda"] = False

        config["amp"] = config["amp"] and config["cuda"]
        config["rank"] = rank if config["cuda"] else 0

        # resolve content
        for k, v in config.items():
            if "_target_" not in str(v):
                config[k] = parser.get_parsed_content(k)

        return parser, config

    def checkpoint_save(self, ckpt : str, model : torch.nn.Module, **kwargs):

        save_time = time.time()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({"state_dict": state_dict, **kwargs}, ckpt)

        save_time = time.time() - save_time
        print("Saving checkpoint process:", ckpt, kwargs, "save_time {:.2f}s".format(save_time))

        return save_time

    def checkpoint_load(self, ckpt : str, model : torch.nn.Module, **kwargs):

        if not os.path.isfile(ckpt):
            if self.rank == 0:
                raise ValueError("=> no checkpoint found at '{}'".format(ckpt))
        else:
            checkpoint = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            epoch = checkpoint.get("epoch", 0)

            best_metric = checkpoint.get("best_metric", 0)
            print(f"=> loaded checkpoint {ckpt} (epoch {epoch}) (best_metric {best_metric})")


    def get_shared_memory_list(self, length=0):

        mp.current_process().authkey = np.arange(32, dtype=np.uint8).tobytes()
        shl0 = mp.Manager().list([None]*length)

        if self.distributed:
            # to support multi-node training, we need check for a local process group
            is_multinode = False

            if dist.is_torchelastic_launched():
                local_world_size = int(os.getenv("LOCAL_WORLD_SIZE"))
                world_size = int(os.getenv("WORLD_SIZE"))
                group_rank = int(os.getenv("GROUP_RANK"))
                if world_size > local_world_size:
                    is_multinode = True
                    # we're in multi-node, get local world sizes
                    lw = torch.tensor(local_world_size, dtype=torch.int, device=self.device)
                    lw_sizes = [torch.zeros_like(lw) for _ in range(world_size)]
                    dist.all_gather(tensor_list=lw_sizes, tensor=lw)

                    src = g_rank = 0
                    while src < world_size:
                        shl_list = [shl0]
                        # create sub-groups local to a node, to share memory only within a node
                        # and broadcast shared list within a node
                        group = dist.new_group(ranks=list(range(src, src + local_world_size)))
                        dist.broadcast_object_list(shl_list, src=src, group=group, device=self.device)
                        dist.destroy_process_group(group)
                        if group_rank==g_rank:
                            shl = shl_list[0]
                        src = src + lw_sizes[src].item() #rank of first process in the next node
                        g_rank += 1

            if not is_multinode:
                shl_list = [shl0]
                dist.broadcast_object_list(shl_list, src=0,  device=self.device)
                shl = shl_list[0]

        else:
            shl = shl0

        return shl


    def get_train_loader(self, data, cache_rate=0, persistent_workers=False):

        distributed = self.distributed
        num_workers = self.config["num_workers"]
        batch_size = self.config["batch_size"]

        train_transform = self.data_tranform_builder(augment=True, resample_label=True)
        if cache_rate > 0:
            runtime_cache = self.get_shared_memory_list(length=len(data))
            train_ds = CacheDataset(
                data=data, transform=train_transform, copy_cache=False, cache_rate=cache_rate, runtime_cache=runtime_cache
            )
        else:
            train_ds = Dataset(data=data, transform=train_transform)

        train_sampler = DistributedSampler(train_ds, shuffle=False) if distributed else None
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            sampler=train_sampler,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=True,
        )

        return train_loader

    def get_val_loader(self, data, cache_rate=0, resample_label=False, persistent_workers=False):

        distributed = self.distributed
        num_workers = self.config["num_workers"]

        val_transform = self.data_tranform_builder(augment=False, resample_label=resample_label)

        if cache_rate > 0:
            runtime_cache = self.get_shared_memory_list(length=len(data))
            val_ds = CacheDataset(
                data=data, transform=val_transform, copy_cache=False, cache_rate=cache_rate, runtime_cache=runtime_cache
            )
        else:
            val_ds = Dataset(data=data, transform=val_transform)

        val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            sampler=val_sampler,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=True,
        )

        return val_loader

    def train(self):

        if self.rank == 0:
            print("Segmenter train called")

        if self.loss_function is None:
            raise ValueError("CONFIG loss function is not provided")
        if self.acc_function is None:
            raise ValueError("CONFIG accuracy function is not provided")

        config = self.config
        model = self.model
        rank = self.rank
        distributed = self.distributed
        sliding_inferrer = self.sliding_inferrer

        loss_function = self.loss_function
        acc_function = self.acc_function
        grad_scaler = self.grad_scaler

        num_epochs = config["num_epochs"]
        use_amp = config["amp"]
        use_cuda = config["cuda"]
        ckpt_path = config["ckpt_path"]
        sigmoid = config["sigmoid"]

        if config.get("validation_key", None) is not None:
            train_files, _ = datafold_read(datalist=config["data_list_file_path"], basedir=config["data_file_base_dir"], fold=-1)
            validation_files, _ = datafold_read(datalist=config["data_list_file_path"], basedir=config["data_file_base_dir"], fold=-1, key=config["validation_key"])
        else:
            train_files, validation_files = datafold_read(datalist=config["data_list_file_path"], basedir=config["data_file_base_dir"], fold=config["fold"])
            
        if config["quick"]:  # quick run on a smaller subset of files
            train_files, validation_files = train_files[:8], validation_files[:8]
        if self.rank == 0:
            print("train_files files", len(train_files), "validation files", len(validation_files))

        if len(validation_files)==0:
            warnings.warn("No validation files found!")

        cache_rate_train, cache_rate_val = self.get_cache_rate(
            train_cases=len(train_files), validation_cases=len(validation_files)
        )

        # train loader
        train_loader = self.get_train_loader(data=train_files, cache_rate=cache_rate_train, persistent_workers=True)
        # validation loader 
        val_loader = self.get_val_loader(
            data=validation_files, cache_rate=cache_rate_val, resample_label=True, persistent_workers=True
        )

        if self.optimizer is None:
            optimizer_part = self.parser.get_parsed_content("optimizer", instantiate=False)
            optimizer = optimizer_part.instantiate(params=model.parameters())
        else:
            optimizer = self.optimizer

        if self.lr_scheduler is None:
            lr_scheduler = WarmupCosineSchedule(optimizer=optimizer, warmup_steps=config["num_warmup_epochs"], warmup_multiplier=0.1, t_total=num_epochs, end_lr=1.0e-6)
        else:
            lr_scheduler = self.lr_scheduler

        tb_writer = None
        csv_path = progress_path = None

        if rank == 0 and ckpt_path is not None:
            # rank 0 is responsible for heavy lifting of logging/saving
            progress_path = os.path.join(ckpt_path, "progress.yaml")

            csv_path = os.path.join(ckpt_path, "accuracy_history.csv")
            self.save_history_csv(
                csv_path=csv_path,
                header=[
                    "epoch",
                    "TP",
                    "FN",
                    "FP",
                    "F1_interim",
                    "DSC_baseline",
                    "loss",
                    "iter",
                    "time",
                    "train_time",
                    "validation_time",
                    "epoch_time"
                ]
            )

        best_ckpt_path = intermediate_ckpt_path = None
        do_torch_save = (rank == 0) and ckpt_path is not None and config["ckpt_save"]
        if do_torch_save:
            best_ckpt_path = os.path.join(ckpt_path, "model.pt")
            intermediate_ckpt_path = os.path.join(ckpt_path, "model_final.pt")

        best_metric = -1
        best_metric_epoch = -1
        pre_loop_time = time.time()

        for epoch in range(num_epochs): 

            if distributed:
                if isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
                dist.barrier()

            epoch_time = start_time = time.time()
            train_loss, train_acc = self.train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_function=loss_function,
                acc_function=acc_function,
                grad_scaler=grad_scaler,
                epoch=epoch,
                rank=rank,
                num_epochs=num_epochs,
                sigmoid=sigmoid,
                use_amp=use_amp,
                use_cuda=use_cuda,
            )
            train_time = "{:.2f}s".format(time.time() - start_time)

            if rank == 0:
                print(
                    "Final training  {}/{}".format(epoch, num_epochs - 1),
                    "loss: {:.4f}".format(train_loss),
                    "acc_avg: {:.4f}".format(np.mean(train_acc)),
                    "acc",
                    train_acc,
                    "time",
                    train_time,
                )

            if lr_scheduler is not None:
                lr_scheduler.step()

            # validate every num_epochs_per_validation epochs (defaults to 1, every epoch)
            val_acc_mean1 = -1
            val_acc_mean2 = -1

            if epoch % config["num_epochs_per_validation"] == 0: 
                start_time = time.time()
                val_loss, avg_TP, avg_FN, avg_FP, val_acc1, val_acc2 = self.val_epoch(
                    model=model,
                    val_loader=val_loader,
                    sliding_inferrer=sliding_inferrer,
                    loss_function=loss_function,
                    acc_function=acc_function,
                    epoch=epoch,
                    rank=rank,
                    num_epochs=num_epochs,
                    sigmoid=sigmoid,
                    use_amp=use_amp,
                    use_cuda=use_cuda,
                    save_prob=False,
                )

                validation_time = "{:.2f}s".format(time.time() - start_time)

                val_acc_mean1 = float(val_acc1) #float(np.mean(val_acc))
                val_tp = avg_TP
                val_fn = avg_FN
                val_fp = avg_FP
                val_acc_mean2 = float(np.mean(val_acc2))
                if rank == 0:
                    print(
                        "Final validation  {}/{}".format(epoch, num_epochs - 1),
                        "loss: {:.4f}".format(val_loss),
                        "acc_avg (interim): {:.4f}".format(val_acc_mean1),
                        "acc_avg (baseline): {:.4f}".format(val_acc_mean2),
                        "time",
                        validation_time,
                    )

                    timing_dict = dict(
                        train_time=train_time,
                        validation_time=validation_time,
                        epoch_time="{:.2f}s".format(time.time() - epoch_time)
                    )

                    if val_acc_mean1 > best_metric:
                        print(f"New best metric ({best_metric:.6f} --> {val_acc_mean1:.6f}). ")
                        best_metric, best_metric_epoch = val_acc_mean1, epoch
                        save_time = 0
                        if do_torch_save:
                            save_time = self.checkpoint_save(ckpt=best_ckpt_path, model=self.model, epoch=best_metric_epoch, \
                                                            best_metric_interim=val_acc_mean1, best_metric_baseline=val_acc_mean2)
                        if progress_path is not None:
                            self.save_progress_yaml(
                                progress_path=progress_path,
                                ckpt=best_ckpt_path,
                                best_avg_dice_score_epoch=best_metric_epoch,
                                best_avg_dice_score=best_metric,
                                save_time = save_time,
                                **timing_dict,
                            )

                    if csv_path is not None:
                        self.save_history_csv(
                            csv_path=csv_path,
                            epoch=epoch,
                            TP = "{:.2f}".format(val_tp*len(val_loader.dataset)),
                            FN = "{:.2f}".format(val_fn*len(val_loader.dataset)),
                            FP = "{:.2f}".format(val_fp*len(val_loader.dataset)),
                            F1_interim="{:.3f}".format(val_acc_mean1),
                            DSC_baseline="{:.3f}".format(val_acc_mean2),
                            loss="{:.3f}".format(val_loss),
                            iter=epoch * len(train_loader.dataset),
                            time="{:.2f}s".format(time.time() - pre_loop_time),
                            **timing_dict,
                        )

                if do_torch_save:
                    self.checkpoint_save(ckpt=intermediate_ckpt_path.replace('final.pt', f'{epoch}.pt'), model=self.model, epoch=epoch)

        #### end of main epoch loop
        train_loader = None
        val_loader = None


    def infer(self, testing_files=None):

        output_path = self.config["infer"].get("output_path", None)
        testing_key = self.config["infer"].get("data_list_key", "testing")

        if output_path is None:
            if self.rank == 0:
                print("Inference output_path is not specified")
            return

        if testing_files is None:
            testing_files, _ = datafold_read(
                datalist=self.config["datalist"],
                basedir=self.config["dataroot"],
                fold=-1,
                key=testing_key,
            )# replace data_list_file_path with datalist


        if self.rank==0:
            print("testing_files files", len(testing_files))
        if len(testing_files)==0:
            warnings.warn("No testing_files files found!")
            return

        inf_loader = self.get_val_loader(data=testing_files, resample_label=False)
        inf_transform = inf_loader.dataset.transform

        post_transforms1 = DataTransformBuilder.get_postprocess_transform(
            save_mask=True,
            invert=True,
            transform=inf_transform,
            sigmoid=self.config["sigmoid"],
            output_path=output_path,
            save_prob=True, 
            output_postfix="interim",
        )
        post_transforms2 = DataTransformBuilder.get_postprocess_transform(
            save_mask=True,
            invert=True,
            transform=inf_transform,
            sigmoid=self.config["sigmoid"],
            output_path=output_path,
            save_prob=True, 
            output_postfix="baseline",
        )

        start_time = time.time()
        self.val_epoch(
            model=self.model,
            val_loader=inf_loader,
            sliding_inferrer=self.sliding_inferrer,
            rank=self.rank,
            sigmoid=self.config["sigmoid"],
            use_amp=self.config["amp"],
            use_cuda=self.config["cuda"],
            post_transforms=[post_transforms1, post_transforms2],
            save_prob=True, 
        )

        if self.rank == 0:
            print("Inference complete, time {:.2f}s".format(time.time() - start_time))


    def train_epoch(
        self,
        model,
        train_loader,
        optimizer,
        loss_function,
        acc_function,
        grad_scaler,
        epoch,
        rank,
        num_epochs=0,
        sigmoid=False,
        use_amp=True,
        use_cuda=True,
    ):

        model.train()
        device = torch.device(rank) if use_cuda else torch.device("cpu")

        run_loss = CumulativeAverage()
        run_acc = CumulativeAverage()

        start_time = time.time()
        avg_loss = avg_acc = 0

        accum_iter = 1 
        optimizer.zero_grad(set_to_none=True)

        for idx, batch_data in enumerate(train_loader):

            data = batch_data["image"].as_subclass(torch.Tensor).to(device=device)
            target = batch_data["label"].as_subclass(torch.Tensor)
            target1 = torch.zeros_like(target)
            target2 = torch.zeros_like(target)

            # interim 
            target1[target==1] = 1
            target1[target==3] = 1
       
            # baseline
            target2[target==2] = 1
            target2[target==3] = 1
         
            # move to device
            target1 = target1.to(device=device)
            target2 = target2.to(device=device)
            
            with autocast(enabled=use_amp):
                logits1, logits2 = model(data) # batch, super_vision_heads, class, h, w, d
            
            loss =  0.5  *  loss_function(logits1, target1) + \
                    0.5  *  loss_function(logits2, target2) 

            loss = loss / accum_iter 
            grad_scaler.scale(loss).backward()

            if (idx + 1) % accum_iter == 0 or (idx + 1 == len(train_loader)):
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = data.shape[0]
            run_loss.append(loss*accum_iter, count=batch_size)
            avg_loss = run_loss.aggregate()
       
            if rank == 0:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, num_epochs, idx, len(train_loader)),
                    "loss: {:.4f}".format(avg_loss),
                    "acc", avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
                start_time = time.time()

        optimizer.zero_grad(set_to_none=True)

        return avg_loss, avg_acc

    @torch.no_grad()
    def val_epoch(
        self,
        model,
        val_loader,
        sliding_inferrer,
        loss_function=None,
        acc_function=None,
        epoch=0,
        rank=0,
        num_epochs=0,
        sigmoid=False,
        use_amp=True,
        use_cuda=True,
        post_transforms=None,
        save_prob=False,
    ):

        model.eval()
        device = torch.device(rank) if use_cuda else torch.device("cpu")

        run_loss = CumulativeAverage()
        run_TP = CumulativeAverage()
        run_FP = CumulativeAverage()
        run_FN = CumulativeAverage()
        run_acc2 = CumulativeAverage()

        avg_loss = avg_acc1 = avg_acc2 = 0
        avg_TP = avg_FN = avg_FP = 0
        start_time = time.time()
   
        nonrepeated_data_length = len(val_loader.dataset)
        sampler = val_loader.sampler
        if dist.is_initialized and isinstance(sampler, DistributedSampler) and not sampler.drop_last:
            nonrepeated_data_length = len(range(sampler.rank, len(sampler.dataset), sampler.num_replicas))

        for idx, batch_data in enumerate(val_loader):
            
            data = batch_data["image"].as_subclass(torch.Tensor).to(device=device)

            with autocast(enabled=use_amp):
                logits1, logits2 = sliding_inferrer(inputs=data, network=model)
            
            logits1 = logits1.float()
            logits2 = logits2.float()
            pred1 = logits2pred(logits1, sigmoid=sigmoid)
            pred2 = logits2pred(logits2, sigmoid=sigmoid)
            if save_prob:
                pred1 = pred1[:,1:,::] # 0: bg, 1: lesion 
                pred2 = pred2[:,1:,::] # 0: bg, 1: lesion
            
            if isinstance(post_transforms, list):
                batch_data["pred1"] = convert_to_dst_type(pred1, batch_data["image"], dtype=pred1.dtype, device=pred1.device)[0]  # make Meta tensor
                batch_data["pred2"] = convert_to_dst_type(pred2, batch_data["image"], dtype=pred1.dtype, device=pred1.device)[0]  # make Meta tensor
                batch_data["pred"] = batch_data["pred1"]
                pred1 = torch.stack([post_transforms[0](x)["pred"] for x in decollate_batch(batch_data)])

                batch_data["pred"] = batch_data["pred2"]
                pred2 = torch.stack([post_transforms[1](x)["pred"] for x in decollate_batch(batch_data)])
                if pred1.shape != logits1.shape:
                    logits1 = None  # if shape changed due to inverse resampling on un-cropping
                if pred2.shape != logits2.shape:
                    logits2 = None  # if shape changed due to inverse resampling on un-cropping  
            elif post_transforms:
                batch_data["pred1"] = convert_to_dst_type(pred1, batch_data["image"], dtype=pred1.dtype, device=pred1.device)[0]  # make Meta tensor
                batch_data["pred2"] = convert_to_dst_type(pred2, batch_data["image"], dtype=pred1.dtype, device=pred1.device)[0]  # make Meta tensor
                pred1 = torch.stack([post_transforms(x)["pred1"] for x in decollate_batch(batch_data)])
                pred2 = torch.stack([post_transforms(x)["pred2"] for x in decollate_batch(batch_data)])
                if pred1.shape != logits1.shape:
                    logits1 = None  # if shape changed due to inverse resampling on un-cropping
                if pred2.shape != logits2.shape:
                    logits2 = None  # if shape changed due to inverse resampling on un-cropping  
           
            if "label" in batch_data and loss_function is not None and acc_function is not None:
                if idx < nonrepeated_data_length:

                    target = batch_data["label"].as_subclass(torch.Tensor)
                    target1 = torch.zeros_like(target)
                    target2 = torch.zeros_like(target)

                    target1[target==1] = 1 # follow-up
                    target1[target==3] = 1 # follow-up  

                    target2[target==2] = 1 # baseline
                    target2[target==3] = 1 # baseline
                    
                    target1 = target1.to(pred1.device)
                    target2 = target2.to(pred2.device)
                 
                    TP, FP, FN = acc_function[0](pred1, target1) 
                    run_TP.append(torch.tensor(TP, dtype=torch.int64).to(device=device), count=1)
                    run_FP.append(torch.tensor(FP, dtype=torch.int64).to(device=device), count=1)
                    run_FN.append(torch.tensor(FN, dtype=torch.int64).to(device=device), count=1)

                    acc2 = acc_function[1](pred2, target2)
                    batch_size_adjusted = batch_size = data.shape[0]
                    
                    if isinstance(acc2, (list, tuple)):
                        acc2, batch_size_adjusted = acc2
                    run_acc2.append(acc2.to(device=device), count=batch_size_adjusted)

                    if logits1 is not None:
                        #loss = 0.5 * loss_function(logits1, target1) + 0.5 * loss_function(logits2, target2)
                        loss = 1.0 * loss_function(logits1, target1)
                        run_loss.append(loss.to(device=device), count=batch_size)

                avg_loss = run_loss.aggregate()
                avg_TP = run_TP.aggregate() # average 
                avg_FP = run_FP.aggregate()
                avg_FN = run_FN.aggregate()
                avg_acc1 = (2 * avg_TP + 1.0e-5) / (2 * avg_TP + avg_FP + avg_FN + 1.0e-5) # F1 score
                avg_acc1 = round(avg_acc1, 4)

                avg_acc2 = run_acc2.aggregate()

                if rank == 0:
                    print(
                        "Val {}/{} {}/{}".format(epoch, num_epochs, idx, len(val_loader)),
                        "loss: {:.4f}".format(avg_loss),
                        "TP", avg_TP,
                        "FN", avg_FN,
                        "FP", avg_FP,
                        "F1 (interim)", avg_acc1,
                        "DSC (baseline)", avg_acc2,
                        "time {:.2f}s".format(time.time() - start_time),
                    )

            else:
                if rank == 0:
                    print(
                        "Val {}/{} {}/{}".format(epoch, num_epochs, idx, len(val_loader)),
                        "time {:.2f}s".format(time.time() - start_time),
                    )

            start_time = time.time()

        return avg_loss, avg_TP, avg_FN, avg_FP, avg_acc1, avg_acc2

    def get_cache_rate(self, train_cases=0, validation_cases=0, prioritise_train=True):

        config = self.config
        cache_rate = config.get("cache_rate", None)
        total_cases = train_cases + validation_cases
        avail_memory = 0.01 #psutil.virtual_memory().available

        if cache_rate is None:
            cache_rate = 0.0
            image_size = config.get("image_size", None)

            if image_size is not None:
                approx_cache_required = (4 * 2 * np.prod(image_size) * config["input_channels"]) * total_cases
                cache_rate = min(0.5 * avail_memory / float(approx_cache_required), 1.0)
                if cache_rate < 0.1:
                    cache_rate = 0.0  # don't cache small amounts

                if self.rank == 0:
                    print(
                        f"Calculating cache required {approx_cache_required /1024**3:.0f}GB, available RAM {avail_memory/1024**3:.0f}GB given avg image size {image_size}."
                    )
                    if cache_rate < 1:
                        print(
                            f"Available RAM is not enought to cache full dataset, caching a fraction {cache_rate:.2f}"
                        )
                    else:
                        print("Caching full dataset in RAM")
        else:
            if self.rank == 0:
                print(f"Using user specified cache_rate {cache_rate} to cache data in RAM")

        # allocate cache_rate to training files first
        cache_rate_train = cache_rate_val = cache_rate

        if prioritise_train:
            if cache_rate > 0 and cache_rate < 1:
                cache_num = cache_rate * total_cases
                cache_rate_train = min(1.0, cache_num / train_cases) if train_cases > 0 else 0
                if (cache_rate_train < 1 and train_cases > 0) or validation_cases == 0:
                    cache_rate_val = 0
                else:
                    cache_rate_val = (cache_num - cache_rate_train * train_cases) / validation_cases

                if self.rank == 0:
                    print(f"Prioritizing cache_rate training {cache_rate_train} validation {cache_rate_val}")

        return cache_rate_train, cache_rate_val

    def save_history_csv(self, csv_path=None, header=None, **kwargs):
        if csv_path is not None:
            if header is not None:
                with open(csv_path, "w") as myfile:
                    wrtr = csv.writer(myfile, delimiter="\t") 
                    wrtr.writerow(header)
            if len(kwargs):
                with open(csv_path, "a") as myfile:
                    wrtr = csv.writer(myfile, delimiter="\t") 
                    wrtr.writerow(list(kwargs.values()))
        
        df = pd.read_csv(csv_path, sep="\t")
        # then to_excel method converting the .csv file to .xlsx file.
        df.to_excel(csv_path.replace(".csv", ".xlsx"), index=False)
 
    def save_progress_yaml(self, progress_path=None, ckpt=None, **report):

        if ckpt is not None:
            report["model"] = ckpt

        report["date"] = str(datetime.now())[:19]

        if progress_path is not None:
            yaml.add_representer(
                float, lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:float", "{0:.4f}".format(value))
            )
            with open(progress_path, "a") as progress_file:
                yaml.dump([report], stream=progress_file, allow_unicode=True, default_flow_style=None, sort_keys=False)

        print("Progress:", ",".join(f" {k}: {v}" for k, v in report.items()))

    def run(self):
        if self.config["validate"]["enabled"]:
            self.validate()
        elif self.config["infer"]["enabled"]:
            self.infer()
        else:
            self.train()


def run_segmenter_worker(rank=0, config_file: Optional[Union[str, Sequence[str]]] = None, override: Dict = {}):

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    dist_available = dist.is_available()

    if dist_available:
        mgpu = override.get("mgpu", None)
        if mgpu is not None:
            dist.init_process_group(backend="nccl", rank=rank, **mgpu)  # we spawn this process
            mgpu["rank"] = rank
            if rank == 0:
                print("Distributed: initializing multi-gpu tcp:// process group", mgpu)

        elif dist.is_torchelastic_launched():

            rank = int(os.getenv("LOCAL_RANK"))
            world_size = int(os.getenv("LOCAL_WORLD_SIZE"))
            dist.init_process_group(backend="nccl", init_method="env://")  # torchrun spawned it
            override["mgpu"] = {"world_size": world_size, "rank": rank}
            print("Distributed: initializing multi-gpu env:// process group", override["mgpu"])

    segmenter = Segmenter(config_file=config_file, config_dict=override, rank=rank)
    best_metric = segmenter.run()
    segmenter = None

    if dist_available and dist.is_initialized():
        dist.destroy_process_group()

    return best_metric


def run_segmenter(config_file: Optional[Union[str, Sequence[str]]] = None, **kwargs):
    """
    if multiple gpu available, start multiprocessing for all gpus
    """
    nprocs = torch.cuda.device_count()
    if nprocs > 1 and not dist.is_torchelastic_launched():
        kwargs["mgpu"] = {"world_size": nprocs, "init_method": kwargs.get("init_method", "tcp://127.0.0.1:23456")}
        torch.multiprocessing.spawn(run_segmenter_worker, nprocs=nprocs, args=(config_file, kwargs))
    else:
        run_segmenter_worker(0, config_file, kwargs)


if __name__ == "__main__":

    fire, fire_is_imported = optional_import("fire")
    if fire_is_imported:
        fire.Fire(run_segmenter)
    else:
        warnings.warn("Fire commandline parser cannot be imported, using options from config/hyper_parameters.yaml")
        run_segmenter(config_file="config/hyper_parameters.yaml")
