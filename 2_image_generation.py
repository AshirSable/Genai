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
