import numpy as np
from scipy.ndimage import gaussian_filter

def weight_bandwidth(w, return_peak=False):
    '''
    calculate the bandwidth of the weights
    params:
        w: the weights, 1d array
    '''
    # smooth the weights
    # w = gaussian_filter(w, sigma=3, truncate=10)
    # w = np.sign(w[np.argmax(np.abs(w))]) * w
    # interpolate the weights
    x = np.arange(len(w))
    nx = len(w)
    nxnew = 2*nx
    xnew = np.linspace(0, len(w)-1, nxnew)
    w = np.interp(xnew, x, w)
    w = gaussian_filter(w, sigma=3, truncate=10)
    # w = np.sign(w[np.argmax(np.abs(w))]) * w
    # find the peak of the weights
    peak_idx = np.argmax(w)
    # find the left and right half width at half maximum, which is closest to the peak
    left_idx = np.argmin(np.abs(w[:peak_idx] - w[peak_idx]/2)) if peak_idx > 0 else 0
    # find the closest left_idx to the peak_idx
    if isinstance(left_idx, np.ndarray):
        left_idx = left_idx[np.argmin(np.abs(left_idx - peak_idx))]
    right_idx = (np.argmin(np.abs(w[peak_idx:] - w[peak_idx]/2)) + peak_idx) if peak_idx < len(w) else len(w)
    # find the closest right_idx to the peak_idx
    if isinstance(right_idx, np.ndarray):
        right_idx = right_idx[np.argmin(np.abs(right_idx - peak_idx))]
    # calculate the bandwidth
    bandwidth = (right_idx - left_idx) * nx / nxnew
    centerpos = (right_idx + left_idx) * nx / nxnew / 2
    if return_peak:
        return bandwidth, centerpos
    return bandwidth
