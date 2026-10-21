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
