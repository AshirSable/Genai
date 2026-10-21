# ======================================================================
# 1 GAN
# ======================================================================

# %% Step 1: Imports and seeds
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

tf.random.set_seed(42)
rng = np.random.default_rng(42)
print("TensorFlow:", tf.__version__)

# %% Step 2: The real data (target)
def real_batch(n):
    angles = rng.uniform(0, 2 * np.pi, n)
    r = 0.8 + 0.05 * rng.normal(size=n)
    x = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    return x.astype("float32")

sample = real_batch(500)
print("sample shape:", sample.shape)

plt.figure(figsize=(4, 4))
plt.scatter(sample[:, 0], sample[:, 1], s=8, color="tab:blue")
plt.title("Real data (target)")
plt.xlim(-1.3, 1.3)
plt.ylim(-1.3, 1.3)
plt.gca().set_aspect("equal")
plt.show()

# %% Step 3: Build G and D
LATENT_DIM = 2
DATA_DIM = 2
HIDDEN = 32

def build_generator():
    return keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(HIDDEN, activation="leaky_relu"),
        layers.Dense(HIDDEN, activation="leaky_relu"),
        layers.Dense(DATA_DIM, activation="tanh"),
    ], name="generator")

def build_discriminator():
    return keras.Sequential([
        layers.Input(shape=(DATA_DIM,)),
        layers.Dense(HIDDEN, activation="leaky_relu"),
        layers.Dense(HIDDEN, activation="leaky_relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="discriminator")

G = build_generator()
D = build_discriminator()
G.summary()
D.summary()

# %% Step 4: Loss and optimizers
bce = keras.losses.BinaryCrossentropy()
g_optimizer = keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5)
d_optimizer = keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5)

# %% Step 5: Look at the untrained generator
untrained_fake = G(tf.random.normal((500, LATENT_DIM)), training=False).numpy()

plt.figure(figsize=(4, 4))
plt.scatter(sample[:, 0], sample[:, 1], s=8, color="tab:blue", alpha=0.4, label="real")
plt.scatter(untrained_fake[:, 0], untrained_fake[:, 1], s=8, color="tab:red", alpha=0.6, label="fake")
plt.legend()
plt.title("Before training")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_aspect("equal")
plt.show()

# %% Step 6: One training step
BATCH = 128

@tf.function
def train_step(real):
    # Phase A: train the Discriminator
    z = tf.random.normal((BATCH, LATENT_DIM))
    with tf.GradientTape() as d_tape:
        fake = G(z, training=True)
        d_real_pred = D(real, training=True)
        d_fake_pred = D(fake, training=True)
        d_loss = (bce(tf.ones_like(d_real_pred), d_real_pred)
                  + bce(tf.zeros_like(d_fake_pred), d_fake_pred))
    d_grads = d_tape.gradient(d_loss, D.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, D.trainable_variables))

    # Phase B: train the Generator
    z2 = tf.random.normal((BATCH, LATENT_DIM))
    with tf.GradientTape() as g_tape:
        fake2 = G(z2, training=True)
        d_fake2_pred = D(fake2, training=True)
        g_loss = bce(tf.ones_like(d_fake2_pred), d_fake2_pred)
    g_grads = g_tape.gradient(g_loss, G.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, G.trainable_variables))

    return d_loss, g_loss

# %% Step 7: Train
EPOCHS = 4000
SNAPSHOT_EVERY = 500

d_losses, g_losses = [], []
snapshots = {}

for epoch in range(EPOCHS + 1):
    real = tf.convert_to_tensor(real_batch(BATCH))
    d_loss, g_loss = train_step(real)
    d_losses.append(float(d_loss))
    g_losses.append(float(g_loss))

    if epoch % SNAPSHOT_EVERY == 0:
        snapshots[epoch] = G(tf.random.normal((300, LATENT_DIM)), training=False).numpy()
        print(f"epoch {epoch:5d}   D_loss = {float(d_loss):.4f}   G_loss = {float(g_loss):.4f}")

print("Training complete.")

# %% Step 8: Pictures of progress
snap_epochs = sorted(snapshots.keys())
cols = 3
rows = int(np.ceil(len(snap_epochs) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
axes = np.array(axes).reshape(-1)

for i, ep in enumerate(snap_epochs):
    ax = axes[i]
    ax.scatter(sample[:, 0], sample[:, 1], s=6, color="tab:blue", alpha=0.3, label="real")
    ax.scatter(snapshots[ep][:, 0], snapshots[ep][:, 1], s=6, color="tab:red", alpha=0.6, label="fake")
    ax.set_title(f"epoch {ep}")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")

axes[0].legend(loc="upper right", fontsize=7)
plt.tight_layout()
plt.show()

# %% Step 9: Final result and loss curves
final_fake = G(tf.random.normal((500, LATENT_DIM)), training=False).numpy()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].scatter(sample[:, 0], sample[:, 1], s=8, color="tab:blue", alpha=0.4, label="real")
axs[0].scatter(final_fake[:, 0], final_fake[:, 1], s=8, color="tab:red", alpha=0.6, label="fake (trained G)")
axs[0].set_title("Real vs Generated (after training)")
axs[0].set_xlim(-1.5, 1.5)
axs[0].set_ylim(-1.5, 1.5)
axs[0].set_aspect("equal")
axs[0].legend()

axs[1].plot(d_losses, label="D loss", color="tab:purple")
axs[1].plot(g_losses, label="G loss", color="tab:orange")
axs[1].axhline(np.log(4), color="tab:purple", linestyle="--", label="D equilibrium (log 4)")
axs[1].axhline(np.log(2), color="tab:orange", linestyle="--", label="G equilibrium (log 2)")
axs[1].set_xlabel("epoch")
axs[1].set_ylabel("loss")
axs[1].set_title("Training loss curves")
axs[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% Step 10: Two check numbers
radius = np.linalg.norm(final_fake, axis=1)
print("mean generated radius:", float(radius.mean()), "(real is about 0.8)")
print("mean D(final_fake):", float(D(final_fake, training=False).numpy().mean()), "(about 0.5 = D is fooled)")


# ======================================================================
# 2 IMAGE GENERATION
# ======================================================================

# %% Step 1: Setup
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import truststore
truststore.inject_into_ssl()

import torch
import matplotlib.pyplot as plt
from IPython.display import display
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# %% Step 2: Load the model and set the knobs
MODEL_ID = "segmind/tiny-sd"
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, dtype=torch.float32)
pipe = pipe.to(device)
inference_steps = 20
guidance_scale = 7.5
print("Loaded:", MODEL_ID)

# %% Step 3: One image
prompt = "A tiny, cute white kitten sleeping inside a teacup, soft lighting, pastel background"
seed = 123
generator = torch.Generator(device=device).manual_seed(seed)
image = pipe(prompt=prompt, num_inference_steps=inference_steps,
             guidance_scale=guidance_scale, generator=generator).images[0]
display(image)
image.save("generated_image.png")

# %% Step 4: Three seeds
seeds = [10, 20, 30]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, s in zip(axes, seeds):
    generator = torch.Generator(device=device).manual_seed(s)
    img = pipe(prompt=prompt, num_inference_steps=inference_steps,
               guidance_scale=guidance_scale, generator=generator).images[0]
    ax.imshow(img)
    ax.set_title(f"Seed {s}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% Step 5: Simple versus detailed prompt
simple_prompt = "A golden retriever puppy."
detailed_prompt = "A fluffy golden retriever puppy joyfully chasing a red ball in a sunny backyard, vibrant watercolor painting style, warm afternoon sunlight."
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, text, title in zip(axes, [simple_prompt, detailed_prompt], ["Simple", "Detailed"]):
    generator = torch.Generator(device=device).manual_seed(55)
    img = pipe(prompt=text, num_inference_steps=inference_steps,
               guidance_scale=guidance_scale, generator=generator).images[0]
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()


# ======================================================================
# 3 MUSIC GENERATION
# ======================================================================

# %% Step 1: Imports
import truststore
truststore.inject_into_ssl()

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from IPython.display import Audio, display
from transformers import AutoProcessor, MusicgenForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch:", torch.__version__)
print("Device:", device)

# %% Step 2: Load the model
MODEL_ID = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
model = model.to(device)
model.eval()
sample_rate = model.config.audio_encoder.sampling_rate
print("Model loaded:", MODEL_ID)
print("Sampling rate:", sample_rate, "Hz")

# %% Step 3: Generate and play
prompt = "calm ambient piano with soft strings, slow tempo, peaceful evening atmosphere"

inputs = processor(text=[prompt], padding=True, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

torch.manual_seed(42)

with torch.no_grad():
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=256)

audio_musicgen = audio_values[0, 0].detach().cpu().numpy()
display(Audio(audio_musicgen, rate=sample_rate))

# %% Step 4: Save as a wav file
peak = max(np.max(np.abs(audio_musicgen)), 1e-9)
audio_normalized = audio_musicgen / peak
audio_int16 = np.int16(audio_normalized * 32767)
write("musicgen_output.wav", sample_rate, audio_int16)
print("Saved: musicgen_output.wav")

# %% Step 5: Draw the waveform
time = np.arange(len(audio_musicgen)) / sample_rate
plt.figure(figsize=(12, 4))
plt.plot(time, audio_musicgen)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("MusicGen Output - Waveform")
plt.grid(alpha=0.25)
plt.show()

# %% Step 6: Compare two prompts
prompts = [
    "relaxing piano music",
    "gentle solo piano, slow tempo, reflective mood, soft room ambience",
]

for text_prompt in prompts:
    print("Prompt:", text_prompt)
    inputs = processor(text=[text_prompt], padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    torch.manual_seed(55)
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=256)
    clip = output[0, 0].detach().cpu().numpy()
    display(Audio(clip, rate=sample_rate))


# ======================================================================
# 4 OBJECT DETECTION
# ======================================================================

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

