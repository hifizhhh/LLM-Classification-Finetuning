import math
import numpy as np
import keras
import matplotlib.pyplot as plt


def get_lr_callback(batch_size=8, mode="cos", epochs=10, plot=False):
    lr_start, lr_max, lr_min = 1.0e-6, 0.6e-6 * batch_size, 1e-6
    lr_ramp_ep, lr_sus_ep, lr_decay = 2, 0, 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        elif mode == "exp":
            lr = (lr_max - lr_min) * lr_decay ** (
                epoch - lr_ramp_ep - lr_sus_ep
            ) + lr_min
        elif mode == "step":
            lr = lr_max * lr_decay ** ((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == "cos":
            decay_total_epochs = epochs - lr_ramp_ep - lr_sus_ep + 3
            decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(
            np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker="o"
        )
        plt.xlabel("epoch")
        plt.ylabel("lr")
        plt.title("Learning Rate Scheduler")
        plt.grid(True)
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
