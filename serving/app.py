from functools import partial
from io import BytesIO
from pathlib import Path
import pytorch_lightning as pl
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch.nn.functional import softmax
from transformers import ViTImageProcessor
import wandb
import time
import json
import psutil


class ClassPredictions(BaseModel):
    predictions: dict[str, float]


app = FastAPI()


def preprocess_image(image: Image.Image) -> torch.tensor:
    return preprocessor([image])["pixel_values"]


def read_imagefile(file: bytes) -> Image.Image:
    return Image.open(BytesIO(file))

def get_size(bytes):
    """Returns size of bytes in a nice format"""
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024


# modeling stuff
model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)
preprocessor = partial(feature_extractor, return_tensors="pt")

package_path = Path(__file__).parent
MODEL_PATH = package_path / "artifacts" / "vis_trans:v0" / "model.ckpt"


def load_model(model_path: str | Path = MODEL_PATH) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model = checkpoint["hyper_parameters"]["model"]
    labels = checkpoint["hyper_parameters"]["label_names"]
    model.eval()  # To set up inference (disable dropout, layernorm, etc.)
    return model, labels


model, labels = load_model()


def predict(x: torch.tensor, labels: list = labels) -> dict:
    logits = model(x).logits
    probas = softmax(logits, dim=1)

    values, indices = torch.topk(probas[0], 5)
    return_dict = {labels[int(i)]: float(v) for i, v in zip(indices, values)}
    return return_dict



# web stuff
@app.get("/")
def get_root() -> dict:
    logger.info("Received request on the root endpoint")
    return {"status": "ok"}


@app.post("/predict", response_model=ClassPredictions)
async def predict_api(file: UploadFile = File(...)) -> ClassPredictions:
    # log timing and network
    logger.info(f"Predict endpoint started timing")
    started_at = time.time()
    io_1 = psutil.net_io_counters()
    bytes_sent, bytes_recv = io_1.bytes_sent, io_1.bytes_recv

    # get file
    logger.info(f"Predict endpoint received image {file.filename}")
    file_extension = file.filename.split(".")[-1]
    valid_extensions = ("jpg", "jpeg", "png")
    if file_extension not in valid_extensions:
        raise TypeError(
            f"File extension for {file.filename} should be one of {valid_extensions}"
        )
    image = read_imagefile(await file.read())

    # process and predict
    logger.info(f"Predict endpoint processed image")
    x = preprocess_image(image)
    predictions = predict(x)

    # finish logging time and network
    logger.info(f"Predict endpoint logging info")
    io_2 = psutil.net_io_counters()
    us, ds = io_2.bytes_sent - bytes_sent, io_2.bytes_recv - bytes_recv
    total_time = time.time() - started_at

    log = {
        "message": f"predictions for {file.filename}: {predictions}",
        "top_class": list(predictions.keys())[0],
        "score": list(predictions.values())[0],
        "latency": total_time,
        "upload_speed": get_size(us / total_time),
        "download_speed": get_size(ds / total_time)
    }
    logger.info(json.dumps(log))
    return ClassPredictions(predictions=predictions)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
