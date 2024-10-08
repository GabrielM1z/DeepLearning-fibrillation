import numpy as np
import scipy.io
import scipy.interpolate


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
            if len(signal) == 0:
                return np.zeros(self.target_length)  # renvoie un tableau de zéros si le signal est vide

            x = np.arange(len(signal))  # indices d'origine
            x_new = np.linspace(0, len(signal) - 1, self.target_length)  # nouveaux indices
            
            interpolator = scipy.interpolate.interp1d(x, signal, kind='linear', fill_value='extrapolate')
            return interpolator(x_new)  # retourne le signal interpolé




    def resize_signal(self, signal, method='padding'):

        if method == 'padding':
            return self.zero_padding(signal)
        elif method == 'truncate':
            return self.truncate_signal(signal)
        elif method == 'interpolate':
            return self.interpolate_signal(signal)
        else:
            raise ValueError(f"Method {method} not recognized. Use 'padding', 'truncate', 'interpolate', or 'segment'.")
