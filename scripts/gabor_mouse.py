import os
import sys
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mouse_id', type=int, default=1, help='mouse id')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--data_path', type=str, default='../data/', help='path to data file')
parser.add_argument('--n_neurons', type=int, default=-1, help='number of neurons, -1 for all')
parser.add_argument('--n_stims', type=int, default=-1, help='number of stimuli, -1 for all')
parser.add_argument('--helper_path', type=str, default='../../../approxineuro/notebooks', help='path to helper file')
parser.set_defaults(normalize=False)
args = parser.parse_args()

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# util_path = os.path.join(parent_dir, 'notebooks')
util_path = args.helper_path
if util_path not in sys.path:
    sys.path.append(util_path)

np.random.seed(args.seed)

# from utils.data import *
from utils import data, metrics
mouse_id = args.mouse_id
data_path = args.data_path

# load neurons
fname = '%s_nat60k_%s.npz'%(data.db[mouse_id]['mname'], data.db[mouse_id]['datexp'])
spks, istim_train, istim_test, xpos, ypos, spks_rep_all = data.load_neurons(file_path = os.path.join(data_path, fname), mouse_id = mouse_id)
n_stim, n_max_neurons = spks.shape

# split train and validation set
itrain, ival = data.split_train_val(istim_train, train_frac=0.9)
ineur = np.arange(0, n_max_neurons) #np.arange(0, n_neurons, 5)

# normalize spks
ntrain = spks.shape[0]
spks, spks_rep_all = data.normalize_spks(spks, spks_rep_all, itrain)

ineurons = np.arange(data.NNs_valid[mouse_id])
fev_test = metrics.fev(spks_rep_all)
isort_neurons = np.argsort(fev_test)[::-1]
ineur = isort_neurons[ineurons]

spks = spks[:,ineur]
spks_rep_all = [spks_rep_all[i][:,ineur] for i in range(len(spks_rep_all))]

img_all = data.load_images(data_path, file=os.path.join(data_path, data.img_file_name[mouse_id]), downsample=2, crop=False)
nimg, Ly, Lx = img_all.shape
print('img: ', img_all.shape, img_all.min(), img_all.max())

n_stim = -1 # spks.shape[0]
n_neurons = -1

# generate random data
if n_stim > 0:
    istims = np.random.choice(spks.shape[0], n_stim, replace=False)
else:
    n_stim = spks.shape[0]
    istims = np.arange(n_stim)
if n_neurons > 0:
    ineurons = np.random.choice(spks.shape[1], n_neurons, replace=False)
    X_test = [spks_rep_all[i][:,ineurons] for i in range(len(spks_rep_all))]
else:
    n_neurons = spks.shape[1]
    ineurons = np.arange(n_neurons)
    X_test = spks_rep_all.copy()

X = spks[istims][:,ineurons]

img = img_all[istim_train][istims].transpose(1,2,0)
img_test = img_all[istim_test].transpose(1,2,0)
print(f'img: {img.shape}, X: {X.shape}')
Ly, Lx, _ = img.shape


from utils import gabor
result_dict = gabor.fit_gabor_model(X, img, X_test, img_test)

# save all gabor parameters
weight_path = os.path.join(parent_dir, 'weights', 'gabor')
save_path = os.path.join(weight_path, f'gabor_params_{data.db[mouse_id]["mname"]}.npz')
if not os.path.exists(weight_path):
    os.makedirs(weight_path)
np.savez(save_path, **result_dict)

# log the results
res_fname = f'gabor_model_{data.db[mouse_id]["mname"]}_result.txt'
with open(res_fname, 'a') as f:
    f.write(f'{save_path}\n')
    f.write(f'FEVE(test)={result_dict["feve"].mean():.4f}\n')