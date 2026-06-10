import numpy as np
import matplotlib.pyplot as plt
from typing import Union

def plot2D(arr: Union[np.ndarray, list], fs: float, 
                                        points = 10000,
                                        title="Frequency Spectrum") -> None:
    if isinstance(arr, list):
        x = np.array(arr)
    else:
        x = arr
    
    a_hat = abs(np.fft.fft(x))
    a_hat = np.fft.fftshift(a_hat)
    
    N = a_hat.size
    f = np.fft.fftfreq(N, d=1/fs)
    f = np.fft.fftshift(f)
    
    plt.plot(f, a_hat, linewidth=0.8)
    plt.title(title, fontsize=16)
    plt.ylabel(r'$\vert\hat{x}(f)\vert$', fontsize=16)
    plt.xlabel('Frequency (Hertz)', fontsize=16)
    
    plt.show()