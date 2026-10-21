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
