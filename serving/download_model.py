import wandb
from pathlib import Path


# package_path = Path(__file__).parent
run = wandb.init()
model_path = wandb.use_artifact("amadorschulze92/Foodformer/vis_trans:production").download()
print(model_path)
run.finish()