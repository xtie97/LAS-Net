import os 
import torch
from monai.apps.auto3dseg import AutoRunner

# specify the algorithm to run
algo = "lasnet" 

runner = AutoRunner(input="./configs/hyper_parameters.yaml", algos=algo, work_dir=f"./train_{algo}", train=True, ensemble=False)

# set the device information
num_gpus = torch.cuda.device_count()
cuda_visible_devices = list(range(num_gpus))
runner.set_device_info(cuda_visible_devices=cuda_visible_devices)

# run the algorithm
runner.set_num_fold(1)  
runner.run()

