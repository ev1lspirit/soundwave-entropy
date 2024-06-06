import os
import scipy.io.wavfile as wav
import numpy as np


def audioread(address: str):
    if not os.path.exists(address):
        raise FileExistsError("No such wav. file: %s" % address)
    base, ext = os.path.splitext(address)
    if ext != '.wav':
        raise TypeError("Invalid extension, expected .wav, got %s" % ext)
    fs_, signal_ = wav.read(address)
    signal_ = np.transpose(signal_)
    if signal_.ndim > 1:
            signal_ = signal_[0]
    return fs_, signal_ / 32767

