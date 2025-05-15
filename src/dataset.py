import tensorflow as tf
import keras
import keras_nlp
import numpy as np
from src.config import CFG

preprocessor = keras_nlp.models.DebertaV3Preprocessor.from_preset(
    CFG.preset, sequence_length=CFG.sequence_length
)


def preprocess_fn(text, label=None):
    text = preprocessor(text)
    return (text, label) if label is not None else text


def build_dataset(texts, labels=None, batch_size=32, cache=True, shuffle=1024):
    AUTO = tf.data.AUTOTUNE
    slices = (
        (texts,)
        if labels is None
        else (texts, keras.utils.to_categorical(labels, num_classes=3))
    )
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.cache() if cache else ds
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTO)
    opt = tf.data.Options()
    if shuffle:
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt.experimental_deterministic = False
    ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds
