
import numpy as np
from scipy.stats import zscore

import numpy as np

def feve_nan(spks, spks_pred, multi_repeats=True):
    """
    spks: list of arrays of shape (n_images, n_repeats, n_neurons), different image might repeat different number of times
    spks_pred: array of shape (n_images, n_neurons)
    """
    n_images = len(spks)

    # calculate total variance of each neuron, ignoring NaNs
    total_var = np.nanvar(np.vstack(spks), axis=0, ddof=1) # shape (n_neurons,)

    # calculate noise variance and variance explained of each neuron
    mse = []
    noise_var = []
    for i in range(n_images):
        # Compute MSE, ignoring NaNs
        mse.append(np.nanmean((spks[i] - spks_pred[i])**2, axis=0))  # shape (n_neurons,)
        noise_var.append(np.nanvar(spks[i], axis=0, ddof=1))  # shape (n_neurons,)
    mse = np.vstack(mse).mean(axis=0)  # shape (n_neurons,)
    noise_var = np.vstack(noise_var).mean(axis=0)  # shape (n_neurons,)

    if not multi_repeats:
        noise_var = 0

    # Calculate explainable variance of each neuron, ignoring NaNs in total variance and noise variance
    fev = (total_var - noise_var) / total_var  # shape (n_neurons,)

    # Calculate explainable variance explained of each neuron, handling NaNs
    feve = 1 - (mse - noise_var) / (total_var - noise_var)  # shape (n_neurons,)
    feve = np.nan_to_num(feve, nan=0)  # Set NaNs in feve to 0, as they are not explainable.

    return fev, feve

def fev_nan(spks):
    """
    spks: list of arrays of shape (n_images, n_repeats, n_neurons), different image might repeat different number of times
    """
    n_images = len(spks)

    # calculate total variance of each neuron, ignoring NaNs
    total_var = np.nanvar(np.vstack(spks), axis=0, ddof=1) # shape (n_neurons,)

    # calculate noise variance and variance explained of each neuron
    noise_var = []
    for i in range(n_images):
        noise_var.append(np.nanvar(spks[i], axis=0, ddof=1))
    noise_var = np.vstack(noise_var).mean(axis=0) # shape (n_neurons,)
    
    # calculate explainable variance of each neuron, ignoring NaNs in total variance and noise variance
    fev = (total_var - noise_var) / total_var # shape (n_neurons,)

    return fev

def fev(spks):
    """
    spks: list of arrays of shape (n_images, n_repeats, n_neurons), different image might repeat different number of times
    """
    n_images = len(spks)

    # calculate total variance of each neuron
    total_var = np.var(np.vstack(spks), axis=0, ddof=1) # shape (n_neurons,)

    # calculate noise variance and variance explained of each neuron
    noise_var = []
    for i in range(n_images):
        noise_var.append(np.var(spks[i], axis=0, ddof=1))
    noise_var = np.vstack(noise_var).mean(axis=0) # shape (n_neurons,)
    
    # calculate explainable variance of each neuron
    fev = (total_var - noise_var) / total_var # shape (n_neurons,)

    return fev

def feve(spks, spks_pred, multi_repeats=True):
    """
    spks: list of arrays of shape (n_images, n_repeats, n_neurons), different image might repeat different number of times
    spks_pred: array of shape (n_images, n_neurons)
    """
    n_images = len(spks)

    # calculate total variance of each neuron
    total_var = np.var(np.vstack(spks), axis=0, ddof=1) # shape (n_neurons,)

    # calculate noise variance and variance explained of each neuron
    mse = []
    noise_var = []
    for i in range(n_images):
        mse.append((spks[i] - spks_pred[i])**2)
        noise_var.append(np.var(spks[i], axis=0, ddof=1))
    mse = np.vstack(mse).mean(axis=0) # shape (n_neurons,)
    noise_var = np.vstack(noise_var).mean(axis=0) # shape (n_neurons,)

    if not multi_repeats: noise_var = 0
    
    # calculate explainable variance of each neuron
    fev = (total_var - noise_var) / total_var # shape (n_neurons,)

    # calculate explainable variance explained of each neuron
    feve = 1 - (mse - noise_var)/ (total_var - noise_var) # shape (n_neurons,)

    return fev, feve

def monkey_feve(spks, spks_pred, repetitions):
    """
    spks: array of shape (n_repeats, n_images, n_neurons)
    spks_pred: array of shape (n_images, n_neurons)
    repetitions: list of integers, number of repetitions for each neuron
    """

    n_neurons = spks[0].shape[-1]

    # calculate total variance, noise variance and variance explained of each neuron
    mse = []
    noise_var = []
    total_var = []
    for i in range(n_neurons):
        mse.append(np.nanmean((spks[:repetitions[i], :, i] - spks_pred[:, i])**2)) 
        noise_var.append(np.nanmean(np.nanvar(spks[:repetitions[i], :, i], axis=0, ddof=1))) 
        total_var.append(np.nanvar(spks[:repetitions[i], :, i], ddof=1))
    mse = np.array(mse)
    noise_var = np.array(noise_var) # shape (n_neurons,)
    total_var = np.array(total_var)

    # calculate explainable variance of each neuron
    fev = (total_var - noise_var) / total_var # shape (n_neurons,)

    # calculate explainable variance explained of each neuron
    feve = 1 - (mse - noise_var)/ (total_var - noise_var) # shape (n_neurons,)

    return fev, feve

def category_variance(spks):
    '''
    Calculate the category variance.
    spks: np.array, shape (n_category, n_stim)
    '''
    ncat, nstim = spks.shape
    category_mean = spks.mean(axis=1)
    residual_var = np.zeros(ncat)
    for i in range(ncat):
        residual_var[i] = np.sum((spks[i] - category_mean[i]) ** 2) / (nstim - 1)
    residual_var = np.mean(residual_var)
    total_variance = spks.var(ddof=1)
    category_variance = (total_variance - residual_var) / total_variance
    return category_variance 

def category_variance_pairwise(spks, labels, ss=None):
    '''
    Calculate the category variance, assuming the same number of stimuli in each category.
    spks: np.array, shape (n_neuron, n_stim)
    labels: np.array, shape (n_stim,)
    '''
    if ss is not None:
        cats = ss
    else:
        cats = np.unique(labels)
    ncat = len(cats)

    nneuron, nstim = spks.shape
    #  zscore the spikes
    spks = zscore(spks, axis=1)

    category_mean = np.zeros((nneuron, ncat))
    for icat, cat in enumerate(cats):
        category_mean[:, icat] = spks[:, labels == cat].mean(axis=1)

    catvar_all = []
    for icat1, cat1 in enumerate(cats):
        for icat2, cat2 in enumerate(cats):
            if icat1 > icat2:
                spks1 = spks[:, labels == cat1]
                spks2 = spks[:, labels == cat2]
                nstim1 = spks1.shape[1]
                nstim2 = spks2.shape[1]
                total_variance = np.var(np.hstack([spks1, spks2]), axis=1, ddof=1)
                # residual_var1 = np.sum((spks1 - category_mean[:, icat1][:, None]) ** 2, axis=1) / (nstim1 - 1)
                # residual_var2 = np.sum((spks2 - category_mean[:, icat2][:, None]) ** 2, axis=1) / (nstim2 - 1)
                # residual_var = np.mean(np.stack([residual_var1, residual_var2]), axis=0)
                residual_var1 = np.sum((spks1 - category_mean[:, icat1][:, None]) ** 2, axis=1)
                residual_var2 = np.sum((spks2 - category_mean[:, icat2][:, None]) ** 2, axis=1)
                residual_var = (residual_var1 + residual_var2) / (nstim1 + nstim2 - 2)
                category_var = total_variance - residual_var
                category_var[np.abs(category_var) < 0.0001] = 0 # control precision
                # Avoid division by zero by checking if total_variance is zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    category_var = np.where(total_variance != 0, category_var / total_variance, np.nan)
                # print(f'cat1={cat1}, cat2={cat2}, category_var={category_var}, total_variance={total_variance}, residual_var={residual_var}')
                catvar_all.append(category_var)
    catvar_all = np.nanmean(np.stack(catvar_all), axis=0)
    return catvar_all


def add_poisson_noise(predictions, fev_target, init_lam=0.7, idxes=None, N_repeats=10, delta=0.01, return_lam=False):
    '''
    predictions: (N_trials, N_neurons)
    fev_targett(int): the target FEV
    init_lam(float): the initial value of noise intensity
    idxes: the indices of neurons to be added noise
    '''
    if idxes is None:
        idxes = np.arange(predictions.shape[-1])
    predictions = predictions[:, idxes]
    lam = init_lam
    stop = False
    track_lam = []
    track_fev = []
    NT, NN = predictions.shape
    # pnoise = np.random.poisson(size=(N_repeats, NT, NN))
    print(f'target fev={fev_target:.3f}')

    from scipy.stats import zscore
    while not stop:
        noisy_predictions = predictions[np.newaxis].repeat(N_repeats, axis=0) + np.random.poisson(lam, size=(N_repeats, NT, NN))
        noisy_fev = fev(noisy_predictions.transpose(1, 0, 2))
        mean_noisy_fev = noisy_fev.mean()
        print(f'lam={lam:.3f}, mean_noisy_fev={mean_noisy_fev:.3f}')
        if (np.abs(mean_noisy_fev - fev_target) < 0.001):
            stop = True
        elif lam in track_lam:
            max_idx = np.argmin(np.abs(np.array(track_fev) - fev_target))
            lam = track_lam[max_idx]
            noisy_predictions = predictions[np.newaxis].repeat(N_repeats, axis=0) + np.random.poisson(lam, size=(N_repeats, NT, NN))
            noisy_fev = fev(noisy_predictions.transpose(1, 0, 2))
            mean_noisy_fev = noisy_fev.mean()
            stop = True
        elif mean_noisy_fev > fev_target:
            track_lam.append(lam)
            track_fev.append(mean_noisy_fev)
            lam += delta
        else:
            track_lam.append(lam)
            track_fev.append(mean_noisy_fev)
            lam -= delta
    print(f'final lam={lam:.3f}, mean_noisy_fev={mean_noisy_fev:.3f}')
    if return_lam:
        return noisy_predictions, noisy_fev, lam
    return noisy_predictions, noisy_fev