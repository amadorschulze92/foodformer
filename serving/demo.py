import gradio as gr
import requests
from fastapi import FastAPI
from PIL import Image
import torch
from transformers import ViTImageProcessor
from functools import partial
from pathlib import Path

app = FastAPI()

def preprocess_image(image: Image.Image) -> torch.tensor:
    return preprocessor([image])["pixel_values"]

# modeling stuff
model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)
preprocessor = partial(feature_extractor, return_tensors="pt")

package_path = Path(__file__).parent
MODEL_PATH = package_path / "model.ckpt" #/ "artifacts" / "vis_trans_v0" / 


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

# gradio app

def gradio_predict(inp):
    x = preprocess_image(inp)
    predictions = predict(x)
    all_predictions = {list(predictions.keys())[i]: float(list(predictions.values())[i]) for i in range(5)}
    return all_predictions


demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(shape=(224, 224), source="webcam", label="Upload Image or Capture from Webcam"),
    outputs=gr.Label(num_top_classes=5, label="Predicted Class"),
    live=False
).launch()
