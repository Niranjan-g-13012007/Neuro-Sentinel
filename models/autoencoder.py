"""TensorFlow/Keras autoencoder used to learn typical behaviour."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

# Quiet TensorFlow start-up logs and avoid oneDNN float round-off differences
# between runs (helps reproducibility). Must be set before importing TF.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

HIDDEN_DIMS = (8, 4)     # encoder hidden layers (decoder mirrors them)
LATENT_DIM = 2           # size of the compressed representation
VALIDATION_FRACTION = 0.20
EARLY_STOPPING_PATIENCE = 10


@dataclass
class AutoencoderResult:
    model: tf.keras.Model
    encoder: tf.keras.Model
    history: dict[str, list[float]]
    train_idx: np.ndarray
    val_idx: np.ndarray
    epochs_run: int
    best_epoch: int
    layer_sizes: list[int]


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow for approximately reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)  # seeds python, numpy and TF together
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # older TF versions or already-initialised runtime
        pass


def build_autoencoder(n_features: int, hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
                      latent_dim: int = LATENT_DIM):
    """Symmetric dense autoencoder: n -> 8 -> 4 -> 2 -> 4 -> 8 -> n.

    The output layer is linear (not ReLU) because the standardized inputs
    contain negative values that a ReLU output could never reproduce.
    """
    inputs = tf.keras.Input(shape=(n_features,), name="activity_features")
    x = inputs
    for i, units in enumerate(hidden_dims):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"encoder_{i + 1}")(x)
    latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(x)
    x = latent
    for i, units in enumerate(reversed(hidden_dims)):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"decoder_{i + 1}")(x)
    outputs = tf.keras.layers.Dense(n_features, activation="linear", name="reconstruction")(x)

    autoencoder = tf.keras.Model(inputs, outputs, name="neurosentinel_autoencoder")
    encoder = tf.keras.Model(inputs, latent, name="neurosentinel_encoder")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    layer_sizes = [n_features, *hidden_dims, latent_dim, *reversed(hidden_dims), n_features]
    return autoencoder, encoder, layer_sizes


def train_autoencoder(X: np.ndarray, epochs: int = 100, batch_size: int = 32,
                      seed: int = 42) -> AutoencoderResult:
    """Train on 80 % of the data and hold out 20 % for validation.

    The split is made explicitly (rather than via Keras ``validation_split``) so
    the held-out indices are known and can be used later to derive the anomaly
    threshold from *unseen* reconstruction errors.
    """
    set_global_seed(seed)
    X = np.asarray(X, dtype="float32")
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=VALIDATION_FRACTION, random_state=seed)

    model, encoder, layer_sizes = build_autoencoder(X.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    history = model.fit(
        X[train_idx], X[train_idx],
        validation_data=(X[val_idx], X[val_idx]),
        epochs=epochs, batch_size=batch_size, shuffle=True,
        callbacks=[early_stop], verbose=0,
    )
    val_loss = history.history["val_loss"]
    return AutoencoderResult(
        model=model,
        encoder=encoder,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        train_idx=train_idx,
        val_idx=val_idx,
        epochs_run=len(val_loss),
        best_epoch=int(np.argmin(val_loss)) + 1,
        layer_sizes=layer_sizes,
    )


def reconstruction_errors(model: tf.keras.Model, X: np.ndarray):
    """Return (per-activity MSE, per-feature squared error, reconstruction)."""
    X = np.asarray(X, dtype="float32")
    reconstruction = model.predict(X, verbose=0)
    squared_error = (X - reconstruction) ** 2
    return squared_error.mean(axis=1), squared_error, reconstruction


def encode(encoder: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """Compressed (latent) representation of every activity."""
    return encoder.predict(np.asarray(X, dtype="float32"), verbose=0)
