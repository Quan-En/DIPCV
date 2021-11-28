# Reference (1): J. Makhoul (1980)
# Reference (2): https://github.com/zafarrafii/Zaf-Python

import numpy as np

def FDCT(audio_signal, dct_type):
    """
    Compute the discrete cosine transform (DCT) using the fast Fourier transform (FFT).
    Inputs:
        audio_signal: audio signal (window_length,)
        dct_type: DCT type (1, 2, 3, or 4)
    Output:
        audio_dct: audio DCT (number_frequencies,)
    """

    # Check if the DCT type is I, II, III, or IV
    if dct_type == 1:

        # Get the number of samples
        window_length = len(audio_signal)

        # Pre-process the signal to make the DCT-I matrix orthogonal
        # (copy the signal to avoid modifying it outside of the function)
        audio_signal = audio_signal.copy()
        audio_signal[[0, -1]] = audio_signal[[0, -1]] * np.sqrt(2)

        # Compute the DCT-I using the FFT
        audio_dct = np.concatenate((audio_signal, audio_signal[-2:0:-1]))
        audio_dct = np.fft.fft(audio_dct)
        audio_dct = np.real(audio_dct[0:window_length]) / 2

        # Post-process the results to make the DCT-I matrix orthogonal
        audio_dct[[0, -1]] = audio_dct[[0, -1]] / np.sqrt(2)
        audio_dct = audio_dct * np.sqrt(2 / (window_length - 1))

        return audio_dct

    elif dct_type == 2:

        # Get the number of samples
        window_length = len(audio_signal)

        # Compute the DCT-II using the FFT
        audio_dct = np.zeros(4 * window_length)
        audio_dct[1 : 2 * window_length : 2] = audio_signal
        audio_dct[2 * window_length + 1 : 4 * window_length : 2] = audio_signal[::-1]
        audio_dct = np.fft.fft(audio_dct)
        audio_dct = np.real(audio_dct[0:window_length]) / 2

        # Post-process the results to make the DCT-II matrix orthogonal
        audio_dct[0] = audio_dct[0] / np.sqrt(2)
        audio_dct = audio_dct * np.sqrt(2 / window_length)

        return audio_dct

    elif dct_type == 3:

        # Get the number of samples
        window_length = len(audio_signal)

        # Pre-process the signal to make the DCT-III matrix orthogonal
        # (copy the signal to avoid modifying it outside of the function)
        audio_signal = audio_signal.copy()
        audio_signal[0] = audio_signal[0] * np.sqrt(2)

        # Compute the DCT-III using the FFT
        audio_dct = np.zeros(4 * window_length)
        audio_dct[0:window_length] = audio_signal
        audio_dct[window_length + 1 : 2 * window_length + 1] = -audio_signal[::-1]
        audio_dct[2 * window_length + 1 : 3 * window_length] = -audio_signal[1:]
        audio_dct[3 * window_length + 1 : 4 * window_length] = audio_signal[:0:-1]
        audio_dct = np.fft.fft(audio_dct)
        audio_dct = np.real(audio_dct[1 : 2 * window_length : 2]) / 4

        # Post-process the results to make the DCT-III matrix orthogonal
        audio_dct = audio_dct * np.sqrt(2 / window_length)

        return audio_dct

    elif dct_type == 4:

        # Get the number of samples
        window_length = len(audio_signal)

        # Compute the DCT-IV using the FFT
        audio_dct = np.zeros(8 * window_length)
        audio_dct[1 : 2 * window_length : 2] = audio_signal
        audio_dct[2 * window_length + 1 : 4 * window_length : 2] = -audio_signal[::-1]
        audio_dct[4 * window_length + 1 : 6 * window_length : 2] = -audio_signal
        audio_dct[6 * window_length + 1 : 8 * window_length : 2] = audio_signal[::-1]
        audio_dct = np.fft.fft(audio_dct)
        audio_dct = np.real(audio_dct[1 : 2 * window_length : 2]) / 4

        # Post-process the results to make the DCT-IV matrix orthogonal
        audio_dct = audio_dct * np.sqrt(2 / window_length)

        return audio_dct


def FDCT2(image, dct_type=2):
    FDCT_along_row = np.apply_along_axis(lambda x: FDCT(x, dct_type), axis=1, arr=image)
    FDCT_along_col = np.apply_along_axis(lambda x: FDCT(x, dct_type), axis=0, arr=FDCT_along_row)
    return FDCT_along_col

def IFDCT2(image, dct_type=3):
    FDCT_along_row = np.apply_along_axis(lambda x: FDCT(x, dct_type), axis=1, arr=image)
    FDCT_along_col = np.apply_along_axis(lambda x: FDCT(x, dct_type), axis=0, arr=FDCT_along_row)
    return FDCT_along_col
