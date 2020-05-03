"""
faye

"""

import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt


sample_rate, signal = scipy.io.wavfile.read("C:/Users/Faye/Downloads/got_s2e8_die_4u.wav")  # File assumed to be in the same directory
signal = signal[0 : int(3.5 * sample_rate)]  # Keep the first 3.5 seconds

plot.subplot(211)

plot.title('Signal (fig.1) and spectogram (fig.2) of a wav file from Game of Thrones')

plot.plot(signal)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

plot.subplot(212)

plot.specgram(signal,  Fs=sample_rate)

plot.xlabel('Time')

plot.ylabel('Frequency')

plot.show()
