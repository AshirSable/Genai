# %% Step 1: Imports
import truststore
truststore.inject_into_ssl()

import time
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("Device:", device)

# %% Step 2: Get and show the sample photo
SAMPLE_URL = "https://ultralytics.com/images/zidane.jpg"
SAMPLE_FILE = Path("object_detection_Sample.jpg")

if not SAMPLE_FILE.exists():
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_FILE)

image = Image.open(SAMPLE_FILE).convert("RGB")
print("Image size:", image.size)

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.show()

# %% Step 3: Two helper functions
def show_comparison(original, detected, title, figsize=(15, 7)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(detected)
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

def draw_detections(pil_image, boxes, labels, scores, categories, threshold=0.60):
    output = pil_image.copy()
    draw = ImageDraw.Draw(output)
    kept = 0
    for box, label, score in zip(boxes, labels, scores):
        score = float(score)
        if score < threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        class_id = int(label)
        class_name = categories[class_id] if 0 <= class_id < len(categories) else str(class_id)
        text = f"{class_name} {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 2, y1 + 2), text, fill="white")
        kept += 1
    return output, kept

# %% Step 4: Faster R-CNN: load
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

frcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
frcnn_categories = frcnn_weights.meta["categories"]
faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights=frcnn_weights).to(device).eval()
frcnn_preprocess = frcnn_weights.transforms()
print("Faster R-CNN loaded.")

# %% Step 5: Faster R-CNN: run and draw
frcnn_input = frcnn_preprocess(image).to(device)

start = time.perf_counter()
with torch.no_grad():
    frcnn_prediction = faster_rcnn([frcnn_input])[0]
frcnn_time = time.perf_counter() - start

frcnn_detected, frcnn_count = draw_detections(
    image,
    frcnn_prediction["boxes"].cpu(),
    frcnn_prediction["labels"].cpu(),
    frcnn_prediction["scores"].cpu(),
    frcnn_categories,
    threshold=0.60,
)

show_comparison(image, frcnn_detected, "Faster R-CNN Detected Output")
print("Detections shown:", frcnn_count)
print("Inference time  :", round(frcnn_time, 3), "seconds")

# %% Step 6: RetinaNet: load
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

retina_weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
retina_categories = retina_weights.meta["categories"]
retinanet = retinanet_resnet50_fpn_v2(weights=retina_weights).to(device).eval()
retina_preprocess = retina_weights.transforms()
print("RetinaNet loaded.")

# %% Step 7: RetinaNet: run and draw
retina_input = retina_preprocess(image).to(device)

start = time.perf_counter()
with torch.no_grad():
    retina_prediction = retinanet([retina_input])[0]
retina_time = time.perf_counter() - start

retina_detected, retina_count = draw_detections(
    image,
    retina_prediction["boxes"].cpu(),
    retina_prediction["labels"].cpu(),
    retina_prediction["scores"].cpu(),
    retina_categories,
    threshold=0.60,
)

show_comparison(image, retina_detected, "RetinaNet Detected Output")
print("Detections shown:", retina_count)
print("Inference time  :", round(retina_time, 3), "seconds")

# %% Step 8: YOLO: load
from ultralytics import YOLO

try:
    YOLO_MODEL = "yolo26n.pt"
    yolo_model = YOLO(YOLO_MODEL)
except Exception:
    print("Current checkpoint unavailable. Using YOLO11 Nano.")
    YOLO_MODEL = "yolo11n.pt"
    yolo_model = YOLO(YOLO_MODEL)

print("Loaded model:", YOLO_MODEL)

# %% Step 9: YOLO: predict and draw
yolo_start = time.perf_counter()
results = yolo_model.predict(source=image, conf=0.40, verbose=False)
yolo_time = time.perf_counter() - yolo_start

result = results[0]
yolo_bgr = result.plot()          # plot() returns a BGR image
yolo_rgb = yolo_bgr[:, :, ::-1]   # flip BGR -> RGB

show_comparison(image, yolo_rgb, f"YOLO Detected Output - {YOLO_MODEL}")
print("Objects detected:", len(result.boxes))
print("Inference time  :", round(yolo_time, 3), "seconds")
