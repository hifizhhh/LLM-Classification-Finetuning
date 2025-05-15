import keras
import keras_nlp
import tensorflow as tf
import numpy as np
from src.config import CFG
from src.dataset import build_dataset
from src.model import build_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.lr_scheduler import get_lr_callback

# Set seed and policy
keras.utils.set_random_seed(CFG.seed)
if tf.config.list_physical_devices("GPU"):
    keras.mixed_precision.set_global_policy("mixed_float16")

# Load data (you can replace with actual file paths)
import pandas as pd

train_df = pd.read_csv("data/train.csv")
valid_df = pd.read_csv("data/valid.csv")

# Build dataset
train_ds = build_dataset(
    train_df["options"].tolist(),
    train_df["class_label"].tolist(),
    batch_size=CFG.batch_size,
)
valid_ds = build_dataset(
    valid_df["options"].tolist(),
    valid_df["class_label"].tolist(),
    batch_size=CFG.batch_size,
    shuffle=False,
)

# Callbacks
lr_cb = get_lr_callback(CFG.batch_size, mode=CFG.scheduler, epochs=CFG.epochs)
ckpt_cb = ModelCheckpoint(
    "best_model.weights.h5",
    monitor="val_log_loss",
    save_best_only=True,
    save_weights_only=True,
)
early_stop_cb = EarlyStopping(
    monitor="val_log_loss", patience=2, restore_best_weights=True
)

# Build model
model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(5e-6),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
    metrics=[
        keras.metrics.CategoricalCrossentropy(name="log_loss"),
        keras.metrics.CategoricalAccuracy(name="accuracy"),
    ],
)

# Train model
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=CFG.epochs,
    callbacks=[lr_cb, ckpt_cb, early_stop_cb],
)

# Save final model
model.save_weights("final_model.weights.h5")
