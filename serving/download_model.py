import wandb
from pathlib import Path
import os


# package_path = Path(__file__).parent
run = wandb.init()
model_path = wandb.use_artifact("amadorschulze92/Foodformer/vis_trans:production").download()
# change path name so no ':'
print(model_path)
new_path = model_path.replace(':', "_")
if not os.path.exists(new_path):
    os.makedirs(new_path)
os.rename(model_path+'/model.ckpt', new_path+'/model.ckpt')
print(new_path)
run.finish()