import numpy as np
import scipy.io
from scipy import signal


class ECGResizing:

    def __init__(self, target_length=None):
        self.target_length = target_length #longueur cible du signal


    def load_ecg(self, filepath):
        mat = scipy.io.loadmat(filepath)
        return mat['val'][0]


    def zero_padding(self, signal):
        if len(signal) < self.target_length:
            pad_size = self.target_length - len(signal)
            return np.pad(signal, (0, pad_size), 'constant')
        else:
            return signal[:self.target_length]


    def truncate_signal(self, signal):
        return signal[:self.target_length]


    def interpolate_signal(self, signal):
        return signal.resample(signal, self.target_length)


    def resize_signal(self, signal, method='padding'):

        if method == 'padding':
            return self.zero_padding(signal)
        elif method == 'truncate':
            return self.truncate_signal(signal)
        elif method == 'interpolate':
            return self.interpolate_signal(signal)
        else:
            raise ValueError(f"Method {method} not recognized. Use 'padding', 'truncate', 'interpolate', or 'segment'.")
