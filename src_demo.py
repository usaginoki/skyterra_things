import joblib
import torch
from transformers import AutoModel, AutoProcessor
import numpy as np
import os
from PIL import Image
from config_demo import *


class ExperimentalClassifier:
    def __init__(
        self,
        model_name=None,
        texts=None,
        mlp_model_path=None,
        scaler_path=None,
    ):
        if model_name is None:
            model_name = default_model_name
        self.model_name = model_name
        if texts is None:
            texts = default_texts
        self.texts = texts
        if mlp_model_path is None:
            mlp_model_path = default_mlp_model_path
        if scaler_path is None:
            scaler_path = default_scaler_path
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.mlp_model = joblib.load(mlp_model_path)
        self.scaler = joblib.load(scaler_path)

    def extract_features(self, img_path: str) -> np.ndarray:
        image = Image.open(img_path)
        gray_image = image.convert("L")
        still_gray_image = gray_image.convert("RGB")
        inputs = self.processor(
            images=still_gray_image,
            text=self.texts,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embed = outputs.image_embeds.numpy()
            return embed

    def classify_image(self, img_path: str) -> int:
        features = self.extract_features(img_path)
        scaled_features = self.scaler.transform(features)
        prediction = self.mlp_model.predict(scaled_features)
        return prediction
