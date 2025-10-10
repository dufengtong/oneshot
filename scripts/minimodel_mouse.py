import os
import sys
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
util_path = os.path.join(parent_dir, 'notebooks')
if util_path not in sys.path:
    sys.path.append(util_path)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mouse_id', type=int, default=1, help='mouse id')
parser.add_argument('--pretrain_mouse_id', type=int, default=-1, help='pretrain mouse id, -1 for using the same mouse to train conv1, -100 for no pretraining')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--nconv1', type=int, default=128, help='number of convolutional filters in first layer')
parser.add_argument('--nconv2', type=int, default=128, help='number of convolutional filters in second layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of convolutional layers')
parser.add_argument('--data_path', type=str, default='../data/', help='path to data file')
parser.add_argument('--ineuron', type=int, default=1, help='index of neuron')
parser.add_argument('--n_stims', type=int, default=-1, help='number of stimuli, -1 for all')
parser.add_argument('--wc_coef', type=float, default=0.2)
parser.add_argument('--hs_readout', type=float, default=0.003)
parser.add_argument('--l2_readout', type=float, default=0.2)
parser.set_defaults(normalize=False)
args = parser.parse_args()

np.random.seed(args.seed)

if args.pretrain_mouse_id == -1:
    args.pretrain_mouse_id = args.mouse_id

# load data
from utils import data
mouse_id = args.mouse_id
depth_separable = True
pool = True
clamp = True
use_30k = False # use all data recorded (>30k) or only 30k, performance will decrease if use only 30k.
data_path = args.data_path

# load images
from utils.data import img_file_name, db, mouse_names, exp_date
xrange_max = 130
if mouse_id == 5:
    xrange_max = 176
if mouse_id in [10, 11]: crop=False
else: crop=True
img = data.load_images(args.data_path, file=os.path.join(args.data_path, img_file_name[mouse_id]), xrange=[xrange_max-130,xrange_max])
nimg, Ly, Lx = img.shape
print('img: ', img.shape, img.min(), img.max(), img.dtype)

# load neurons
fname = '%s_nat60k_%s.npz'%(db[mouse_id]['mname'], db[mouse_id]['datexp'])
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
from utils import metrics
fev_test = metrics.fev(spks_rep_all)
isort_neurons = np.argsort(fev_test)[::-1]
ineur = [isort_neurons[args.ineuron]]
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
    img_train = torch.from_numpy(img[istim_train][itrain]).to(device).unsqueeze(1) # change :130 to 25:100 
else:
    nstims = args.n_stims
    np.random.seed(args.n_stims *(args.seed+1))
    idxes = np.random.choice(ntrain, nstims, replace=False)
    spks_train = torch.from_numpy(spks[itrain[idxes]][:,ineur])
    print('spks_train: ', spks_train.shape, spks_train.min(), spks_train.max())
    img_train = torch.from_numpy(img[istim_train][itrain[idxes]]).to(device).unsqueeze(1) 

print('img_train: ', img_train.shape, img_train.min(), img_train.max())
input_Ly, input_Lx = img_train.shape[-2:]

# build model
from utils import model_builder
seed = args.seed
nlayers = args.nlayers
nconv1 = args.nconv1
nconv2 = args.nconv2
# suffix = 'fixconv2'
suffix = ''
if args.n_stims != -1:
    suffix = f'nstims_{nstims}'
if args.pretrain_mouse_id > -100:
    # if args.pretrain_mouse_id != mouse_id:
    #     suffix = f'pretrainconv1_{mouse_names[args.pretrain_mouse_id]}_{exp_date[args.pretrain_mouse_id]}'
    if args.pretrain_mouse_id == 5:
        suffix = f'pretrainconv1_{mouse_names[args.pretrain_mouse_id]}_{exp_date[args.pretrain_mouse_id]}_xrange_176'
else:
    if suffix != '': suffix += '_'
    suffix += 'nopretrain'
if xrange_max != 130:
    suffix += f'xrange_{xrange_max}'
model, in_channels = model_builder.build_model(NN=1, n_layers=nlayers, n_conv=nconv1, n_conv_mid=nconv2, pool=pool, depth_separable=depth_separable, Wc_coef=args.wc_coef)
model_name = model_builder.create_model_name(mouse_names[mouse_id], exp_date[mouse_id], ineuron=ineur[0], n_layers=nlayers, in_channels=in_channels, clamp=clamp, seed=seed,hs_readout=args.hs_readout, suffix=suffix)

weight_path = os.path.join(parent_dir, 'weights', 'minimodel', mouse_names[mouse_id])
if not os.path.exists(weight_path):
    os.makedirs(weight_path)
model_path = os.path.join(weight_path, model_name)
print('model path: ', model_path)

# initialize model conv1
if args.pretrain_mouse_id > -100:
    pretrain_mouse_name = mouse_names[args.pretrain_mouse_id]
    if pretrain_mouse_name == 'L1_A1': pretrain_mouse_name = 'l1a1'
    elif pretrain_mouse_name == 'L1_A5': pretrain_mouse_name = 'l1a5'
    pretrain_model_name = f'{pretrain_mouse_name}_{exp_date[args.pretrain_mouse_id]}_2layer_16_320_clamp_sensorium_depthsep_pool.pt'
    if args.pretrain_mouse_id == 5:
        pretrain_model_name = f'{pretrain_mouse_name}_{exp_date[args.pretrain_mouse_id]}_2layer_16_320_clamp_sensorium_depthsep_pool_xrange_176.pt'
    elif args.pretrain_mouse_id == 7:
        pretrain_model_name = f'{pretrain_mouse_name}_{exp_date[args.pretrain_mouse_id]}_2layer_16_320_clamp_sensorium_depthsep_pool_nneurons_2068.pt'
    pretrained_model_path = os.path.join(parent_dir, 'weights', 'fullmodel', mouse_names[args.pretrain_mouse_id], pretrain_model_name)
    pretrained_state_dict = torch.load(pretrained_model_path, map_location=device)
    model.core.features.layer0.conv.weight.data = pretrained_state_dict['core.features.layer0.conv.weight']
    # set the weight fix
    model.core.features.layer0.conv.weight.requires_grad = False
    print('loaded pretrained model', pretrained_model_path)
else:
    print('randomly initialized model')
# load the pc as conv2 spatial weights
# conv2_path = os.path.join(parent_dir, 'weights', 'pcs', f'{mouse_names[mouse_id]}_{exp_date[mouse_id]}_spatial_pcs.npy')
# conv2_weight = torch.from_numpy(np.load(conv2_path)).float()
# nconv2, ks, ks = conv2_weight.shape
# conv2_weight = conv2_weight.view(nconv2, 1, ks, ks)
# model.core.features.layer1.ds_conv.spatial_conv.weight.data = conv2_weight
# model.core.features.layer1.ds_conv.spatial_conv.weight.requires_grad = True

model = model.to(device)

# train model
from utils import model_trainer
if not os.path.exists(model_path):
    best_state_dict = model_trainer.train(model, spks_train, spks_val, img_train, img_val, clamp=clamp, device=device, hs_readout=args.hs_readout, l2_readout=args.l2_readout)
    torch.save(best_state_dict, model_path)
    print('saved model', model_path)
model.load_state_dict(torch.load(model_path))
print('loaded model', model_path)

model.eval()
# test model
test_pred = model_trainer.test_epoch(model, img_test)
print('test_pred: ', test_pred.shape, test_pred.min(), test_pred.max())

from utils import metrics
test_fev, test_feve = metrics.feve(spks_rep_all, test_pred)
print('FEVE (test, all): ', np.mean(test_feve))


res_fname = f'minimodel_{mouse_names[mouse_id]}_result.txt'
if args.n_stims != -1:
    res_fname = f'minimodel_{mouse_names[mouse_id]}_result_vary_n_stims.txt'
if args.pretrain_mouse_id != mouse_id:
    res_fname = f'minimodel_{mouse_names[mouse_id]}_result_reuse_conv1.txt'
with open(res_fname, 'a') as f:
    f.write(f'{model_path}\n')
    f.write(f'FEVE(test)={test_feve.mean()*100:0.4f}%\n')