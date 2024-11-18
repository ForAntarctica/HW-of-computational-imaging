import numpy as np

def operation_xx(gsize):
    delta_xx = np.array([[[1, -2, 1]]], dtype = 'float32')
    xxfft = np.fft.fftn(delta_xx, gsize) * np.conj(np.fft.fftn(delta_xx, gsize))
    return xxfft

def operation_xy(gsize):
    delta_xy = np.array([[[1, -1], [-1, 1]]], dtype = 'float32')
    xyfft = np.fft.fftn(delta_xy, gsize) * np.conj(np.fft.fftn(delta_xy, gsize))
    return xyfft

def operation_yy(gsize):
    delta_yy = np.array([[[1], [-2], [1]]], dtype = 'float32')
    yyfft = np.fft.fftn(delta_yy,gsize) * np.conj(np.fft.fftn(delta_yy, gsize))
    return yyfft