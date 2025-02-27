import os
import sys
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mouse_id', type=int, default=1, help='mouse id')
parser.add_argument('--pretrain_mouse_id', type=int, default=-100, help='use pretrained conv1 or not, -1 for using the same mouse to train conv1')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--nconv1', type=int, default=128, help='number of convolutional filters in first layer')
parser.add_argument('--nconv2', type=int, default=128, help='number of convolutional filters in second layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of convolutional layers')
parser.add_argument('--data_path', type=str, default='../data/', help='path to data file')
parser.add_argument('--n_neurons', type=int, default=-1, help='number of neurons, -1 for all')
parser.add_argument('--n_stims', type=int, default=-1, help='number of stimuli, -1 for all')
parser.add_argument('--weight_decay_core', type=float, default=0.1)
parser.add_argument('--img_downsample', type=int, default=1, help='downsample image, 1 for no downsample, 2 for half size')
parser.add_argument('--conv1_ks', type=int, default=25, help='kernel size of first convolutional layer')
parser.add_argument('--conv2_ks', type=int, default=9, help='kernel size of second convolutional layer')
parser.add_argument('--hs_readout', type=float, default=0.0)
parser.add_argument('--helper_path', type=str, default='../../../approxineuro/notebooks', help='path to helper file')
parser.set_defaults(normalize=False)
args = parser.parse_args()

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# util_path = os.path.join(parent_dir, 'notebooks')
util_path = args.helper_path
if util_path not in sys.path:
    sys.path.append(util_path)

np.random.seed(args.seed)

if args.pretrain_mouse_id == -1:
    args.pretrain_mouse_id = args.mouse_id
    
# load data
from utils import data
mouse_id = args.mouse_id
depth_separable = True
if args.img_downsample == 1: pool = True
else: pool = False
clamp = True
use_30k = False # use all data recorded (>30k) or only 30k, performance will decrease if use only 30k.
data_path = args.data_path

# load images
from utils.data import img_file_name, db, mouse_names, exp_date
if mouse_id == 5:
    xrange_max = 176
elif mouse_id <= 6:
    xrange_max = 130
if mouse_id > 6:
    xrange_max = 150
    img = data.load_images(args.data_path, file=os.path.join(args.data_path, img_file_name[mouse_id]), downsample=args.img_downsample, xrange=[0, 150])
else:
    img = data.load_images(args.data_path, file=os.path.join(args.data_path, img_file_name[mouse_id]), xrange=[xrange_max-130,xrange_max], downsample=args.img_downsample)
nimg, Ly, Lx = img.shape
print('img: ', img.shape, img.min(), img.max(), img.dtype)

# load neurons
fname = '%s_nat30k_%s.npz'%(db[mouse_id]['mname'], db[mouse_id]['datexp'])
spks, istim_train, istim_test, xpos, ypos, spks_rep_all = data.load_neurons(file_path = os.path.join(data_path, fname), mouse_id = mouse_id, fixtrain=use_30k)
n_stim, n_max_neurons = spks.shape
print('spks: ', spks.shape, spks.min(), spks.max())
print('spks_rep_all: ', len(spks_rep_all), spks_rep_all[0].shape)
print('istim_train: ', istim_train.shape, istim_train.min(), istim_train.max())
print('istim_test: ', istim_test.shape, istim_test.min(), istim_test.max())

# split train and validation set
itrain, ival = data.split_train_val(istim_train, train_frac=0.9)
print('itrain: ', itrain.shape, itrain.min(), itrain.max())
print('ival: ', ival.shape, ival.min(), ival.max())

# normalize spks
spks, spks_rep_all = data.normalize_spks(spks, spks_rep_all, itrain)

# calculate FEV
ineur = np.arange(0, n_max_neurons) #np.arange(0, n_neurons, 5)
if args.n_neurons != -1:
    from utils import metrics
    fev_test = metrics.fev(spks_rep_all)
    valid_idxes = np.where(fev_test > 0.15)[0]
    if args.n_neurons >= len(valid_idxes):
        print(f'not enough neurons with FEV > 0.15, using all neurons')
        ineur = valid_idxes
        args.n_neurons = len(valid_idxes)
    else:
        np.random.seed(args.n_neurons*args.seed)
        ineur = np.random.choice(valid_idxes, size=args.n_neurons, replace=False)

spks_train = torch.from_numpy(spks[itrain][:,ineur])
spks_val = torch.from_numpy(spks[ival][:,ineur]) 
spks_rep_all = [spks_rep_all[i][:,ineur] for i in range(len(spks_rep_all))]

print('spks_train: ', spks_train.shape, spks_train.min(), spks_train.max())
print('spks_val: ', spks_val.shape, spks_val.min(), spks_val.max())

img_val = torch.from_numpy(img[istim_train][ival]).to(device).unsqueeze(1)
img_test = torch.from_numpy(img[istim_test]).to(device).unsqueeze(1)

print('img_val: ', img_val.shape, img_val.min(), img_val.max())
print('img_test: ', img_test.shape, img_test.min(), img_test.max())

ntrain = len(itrain)
if (args.n_stims == -1) or (args.n_stims  > ntrain): 
    nstims = ntrain
    img_train = torch.from_numpy(img[istim_train][itrain]).to(device).unsqueeze(1) 
else:
    nstims = args.n_stims
    np.random.seed(args.n_stims *(args.seed+1))
    idxes = np.random.choice(ntrain, nstims, replace=False)
    spks_train = torch.from_numpy(spks[itrain[idxes]][:,ineur])
    print('spks_train: ', spks_train.shape, spks_train.min(), spks_train.max())
    img_train = torch.from_numpy(img[istim_train][itrain[idxes]]).to(device).unsqueeze(1) 

print('img_train: ', img_train.shape, img_train.min(), img_train.max())
input_Ly, input_Lx = img_train.shape[-2:]

train_real_responses = torch.ones_like(spks_train)
val_real_responses = torch.ones_like(spks_val)
# set nans to zero
train_real_responses[torch.isnan(spks_train)] = 0
val_real_responses[torch.isnan(spks_val)] = 0
spks_train[torch.isnan(spks_train)] = 0
spks_val[torch.isnan(spks_val)] = 0

# build model
from utils import model_builder
seed = args.seed
nlayers = args.nlayers
nconv1 = args.nconv1
nconv2 = args.nconv2
suffix = ''
if args.n_neurons != -1:
    suffix = f'nneurons_{args.n_neurons}'
if args.n_stims != -1:
    if suffix != '': suffix += '_'
    suffix += f'nstims_{nstims}'
if args.weight_decay_core != 0.1:
    suffix = f'wdcore_{args.weight_decay_core}'
if xrange_max != 130:
    suffix += f'xrange_{xrange_max}'
if args.img_downsample != 1:
    if suffix != '': suffix += '_'
    suffix += f'downsample_{args.img_downsample}'
if (args.conv1_ks != 25) or (args.conv2_ks != 9):
    if suffix != '': suffix += '_'
    suffix += f'ks_{args.conv1_ks}_{args.conv2_ks}'
if args.pretrain_mouse_id > -100:
    if suffix != '': suffix += '_'
    suffix += f'pretrainconv1_{mouse_names[args.pretrain_mouse_id]}_{exp_date[args.pretrain_mouse_id]}'
model, in_channels = model_builder.build_model(NN=len(ineur), n_layers=nlayers, n_conv=nconv1, n_conv_mid=nconv2, pool=pool, depth_separable=depth_separable, input_Ly=input_Ly, input_Lx=input_Lx, kernel_size=[args.conv1_ks, args.conv2_ks], Wc_coef=args.weight_decay_core)
model_name = model_builder.create_model_name(mouse_names[mouse_id], exp_date[mouse_id], n_layers=nlayers, in_channels=in_channels, clamp=clamp, seed=seed, suffix=suffix, pool=pool,hs_readout=args.hs_readout)

weight_path = os.path.join(parent_dir, 'weights', 'fullmodel', mouse_names[mouse_id])
if not os.path.exists(weight_path):
    os.makedirs(weight_path)
model_path = os.path.join(weight_path, model_name)
print('model path: ', model_path)

# initialize model conv1
if args.pretrain_mouse_id >-100:
    pretrain_mouse_name = mouse_names[args.pretrain_mouse_id]
    if pretrain_mouse_name == 'L1_A1': pretrain_mouse_name = 'l1a1'
    elif pretrain_mouse_name == 'L1_A5': pretrain_mouse_name = 'l1a5'
    pretrain_model_name = f'{pretrain_mouse_name}_{exp_date[args.pretrain_mouse_id]}_2layer_16_320_clamp_sensorium_depthsep_pool.pt'
    if args.pretrain_mouse_id == 5:
        pretrain_model_name = f'{pretrain_mouse_name}_{exp_date[args.pretrain_mouse_id]}_2layer_16_320_clamp_sensorium_depthsep_pool_xrange_176.pt'
    pretrained_model_path = os.path.join(parent_dir, 'weights', 'fullmodel', mouse_names[args.pretrain_mouse_id], pretrain_model_name)
    pretrained_state_dict = torch.load(pretrained_model_path, map_location=device)
    model.core.features.layer0.conv.weight.data = pretrained_state_dict['core.features.layer0.conv.weight']
    # set the weight fix
    model.core.features.layer0.conv.weight.requires_grad = False
    print('loaded pretrained model', pretrained_model_path)

model = model.to(device)

# train model
from utils import model_trainer
if not os.path.exists(model_path):
    best_state_dict = model_trainer.monkey_train(model, spks_train, train_real_responses, spks_val, val_real_responses, img_train, img_val, device=device)
    torch.save(best_state_dict, model_path)
    print('saved model', model_path)
model.load_state_dict(torch.load(model_path))
print('loaded model', model_path)

model.eval()
# test model
test_pred = model_trainer.test_epoch(model, img_test)
print('test_pred: ', test_pred.shape, test_pred.min(), test_pred.max())

from utils import metrics
test_fev, test_feve = metrics.feve_nan(spks_rep_all, test_pred)
print('FEVE (test, all): ', np.mean(test_feve))

threshold = 0.15
print(f'filtering neurons with FEV > {threshold}')
valid_idxes = np.where(test_fev > threshold)[0]
print(f'valid neurons: {len(valid_idxes)} / {len(test_fev)}')
print(f'FEVE (test, FEV>0.15): {np.mean(test_feve[test_fev > threshold])}')


res_fname = f'fullmodel_medial_{mouse_names[mouse_id]}_result.txt'
if args.n_neurons != -1:
    res_fname = f'fullmodel_medial_{mouse_names[mouse_id]}_result_vary_n_neurons.txt'
if args.n_stims != -1:
    res_fname = f'fullmodel_medial_{mouse_names[mouse_id]}_result_vary_n_stims.txt'
with open(res_fname, 'a') as f:
    f.write(f'{model_path}\n')
    f.write(f'FEVE(test)={test_feve[test_fev > threshold].mean()*100:0.4f}%\n')


