import os 
import torch
from monai.apps.auto3dseg import AutoRunner

import os
import shutil
import torch
from monai.apps.auto3dseg import AutoRunner

# create the template path and specify the algorithm to run
templates_path = "algorithm_templates"
algo = "lasnet"
os.makedirs(templates_path, exist_ok=True)
shutil.copytree("scripts", f"algorithm_templates/{algo}/scripts", dirs_exist_ok=True)
shutil.copytree("configs", f"algorithm_templates/{algo}/configs", dirs_exist_ok=True)
shutil.copyfile("algo_object.pkl", f"algorithm_templates/{algo}/algo_object.pkl")

# specify the path to save the code and models
work_dir = f"./train_{algo}"

runner = AutoRunner(
    input=os.path.join(templates_path, algo, "configs", "hyper_parameters.yaml"),
    algos=algo,
    work_dir=work_dir,
    templates_path_or_url=templates_path,
    train=True,
    ensemble=False,
)

# set the device information
num_gpus = torch.cuda.device_count()
cuda_visible_devices = list(range(num_gpus))
runner.set_device_info(cuda_visible_devices=cuda_visible_devices)

# run the algorithm
runner.set_num_fold(1)  
runner.run()

