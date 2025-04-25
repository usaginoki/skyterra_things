from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
import os
import time

# "google/siglip-large-patch16-384"
model_name = "google/siglip-large-patch16-384"
print("Loading model...")
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
print("Model loaded")

class_0_path = "../data/parsed/class_0"
class_1_path = "../data/parsed/class_1"

num_img_per_class = 0
texts = [""]


def extract_features(img_path: str) -> np.ndarray:
    image = Image.open(img_path)
    gray_image = image.convert("L")
    still_gray_image = gray_image.convert("RGB")
    inputs = processor(
        images=still_gray_image, text=texts, padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        embed = outputs.image_embeds.numpy()
        return embed


def load_data(class_path: str, num_img: int = 0):
    if num_img == 0:
        img_paths = os.listdir(class_path)
    else:
        img_paths = os.listdir(class_path)[:num_img]
    features = []
    for img_path in img_paths:
        try:
            features.append(extract_features(os.path.join(class_path, img_path)))
        except Exception as e:
            print(f"Error extracting features from {img_path}: {e}")
            continue
    num_img = len(features)
    labels = [0 if class_path == class_0_path else 1] * num_img
    return np.array(features), np.array(labels)


print("Loading data...")
start_time = time.time()
class_0_features, class_0_labels = load_data(class_0_path, num_img_per_class)
class_1_features, class_1_labels = load_data(class_1_path, num_img_per_class)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

print(class_0_features.shape)
print(class_1_features.shape)

# concatenate the features and labels
X = np.concatenate([class_0_features, class_1_features], axis=0)
y = np.concatenate([class_0_labels, class_1_labels], axis=0)

X = X.reshape(X.shape[0], -1)

# save the data
np.save("siglip2/X.npy", X)
np.save("siglip2/y.npy", y)
