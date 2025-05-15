import keras
import keras_nlp
import tensorflow as tf
from src.config import CFG


def build_model():
    # Define inputs
    inputs = {
        "token_ids": keras.Input(shape=(2, None), dtype=tf.int32, name="token_ids"),
        "padding_mask": keras.Input(
            shape=(2, None), dtype=tf.int32, name="padding_mask"
        ),
    }

    # Backbone
    backbone = keras_nlp.models.DebertaV3Backbone.from_preset(CFG.preset)

    # Encode response A
    response_a = {k: v[:, 0, :] for k, v in inputs.items()}
    embed_a = backbone(response_a)

    # Encode response B
    response_b = {k: v[:, 1, :] for k, v in inputs.items()}
    embed_b = backbone(response_b)

    # Combine
    x = keras.layers.Concatenate(axis=-1)([embed_a, embed_b])
    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(3, activation="softmax", name="classifier")(x)

    return keras.Model(inputs, outputs)
