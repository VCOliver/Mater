from mater import *
from mater import fft
import random
import numpy as np

def main():
    """Main entry point for the mater package."""
    # Signal parameters
    fs = 100  # Sampling frequency
    N = 3000  # Number of samples
    f_cosine = 20  # 20 Hz cosine frequency
    
    # Generate time vector
    t = np.linspace(0, N/fs, N, endpoint=False)
    
    # Generate random noise
    random_floats = [random.uniform(-5, 5) for _ in range(N)]
    
    # Generate 1 kHz cosine signal
    cosine_signal = np.cos(2 * np.pi * f_cosine * t)
    
    # Combine random noise with cosine signal
    combined_signal = np.array(random_floats) + cosine_signal
    
    fft.plot2D(combined_signal, fs)

if __name__ == '__main__':
    main()