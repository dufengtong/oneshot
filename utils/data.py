import os
import cv2
import numpy as np
from scipy.io import loadmat

db = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
db[0]['mname'], db[0]['datexp'], db[0]['blk'] = 'L1_A5', '2023_02_27', '1' # tlag: [3.5, 4.5] nplanes = 2
db[1]['mname'], db[1]['datexp'], db[1]['blk'] = 'L1_A1', '2023_03_06', '1'# tlag: [3.5, 4.5] nplanes = 2
db[2]['mname'], db[2]['datexp'], db[2]['blk'] = 'FX9', '2023_05_15', '2' # [4, 5] nplanes = 4
db[3]['mname'], db[3]['datexp'], db[3]['blk'] = 'FX10', '2023_05_16', '1' # tlags = [4, 5] nplanes = 4
db[4]['mname'], db[4]['datexp'], db[4]['blk'] = 'FX8', '2023_05_16', '2' # tlags = [4, 5] nplanes = 4
db[5]['mname'], db[5]['datexp'], db[5]['blk'] = 'FX20', '2023_09_29', '1' # tlags = [4.5, 5.5] nplanes = 2
db[6]['mname'], db[6]['datexp'], db[6]['blk'] = 'FX40', '2024_10_29', '1' # tlag: [4, 5] nplanes = 4 # no timeline?
db[7]['mname'], db[7]['datexp'], db[7]['blk'] = 'FX41', '2025_01_23', '1' # tlag: [4, 5] nplanes = 4
db[8]['mname'], db[8]['datexp'], db[8]['blk'] = 'FX43', '2025_01_24', '1' # tlag: [4, 5] nplanes = 4
db[9]['mname'], db[9]['datexp'], db[9]['blk'] = 'FX42', '2025_01_25', '1' # tlag: [4, 5] nplanes = 4
db[10]['mname'], db[10]['datexp'], db[10]['blk'] = 'FX41', '2025_02_14', '2' # tlag: [4, 5] nplanes = 4
db[11]['mname'], db[11]['datexp'], db[11]['blk'] = 'FX41', '2025_05_12', '3' # tlag: [4, 5] nplanes = 4
db[12]['mname'], db[12]['datexp'], db[12]['blk'] = 'FX43', '2025_05_19', '4' # tlag: [4, 5] nplanes = 4


mouse_names = ['L1_A5', 'L1_A1',  'FX9', 'FX10', 'FX8', 'FX20', 'FX40', 'FX41', 'FX43', 'FX42', 'FX41', 'FX41', 'FX43']
exp_date = ['022723', '030623', '051523', '051623', '051623', '092923', '102924', '012325', '012425', '012525', '021425', '051225', '051925']
NNs = [6636,6055,3575,4792,5804,2746, 4261,0,0,6049,5247,3491, 4180]
NNs_valid = [4242,2840,926,3040,2217,1239, 0, 2068]
img_file_name = ['nat60k_text16_old.mat', # nat60k images and text16 images are original images
                 'nat60k_text16_old.mat',
                 'nat60k_text16.mat',# nat60k images are original images, text16 images are rezscored to match nat60k mean and std
                 'nat60k_text16.mat',
                 'nat60k_text16.mat',
                 'nat60k_text16.mat',
                 '8x4_nat30k_text16.mat',
                 '8x4_nat30k_text16.mat',
                 '8x4_nat30k_text16.mat',
                 '8x4_nat30k_text16.mat',
                 '8x4_nat30k_text16.mat',
                 '8x4_nat30k_text16.mat',
                 '8x4_nat30k_text16.mat',] # medial area recording

def split_area(mouse_id, xpos, ypos, ineur, retinotopy_path = '/media/carsen/ssd1/github/retinotopy/aligned'):
    x_pixel_ratio = 0.5
    y_pixel_ratio = 0.5
    if mouse_id == 5:
        point1 = [-200, 0]
        point2 = [-500, 600]
    if mouse_id == 7:
        point1 = [-650, 0]
        point2 = [-250, 420]
        x_pixel_ratio = 0.75
    if mouse_id == 8:
        point1 = [-700, 0]
        point2 = [-300, 420]
        x_pixel_ratio = 0.75
    if mouse_id == 10:
        point1 = [-600, 0]
        point2 = [0, 800]
        hmax_angle = 88
        x_pixel_ratio = 0.75
    if mouse_id == 11:
        hmax_angle = 88
        # x_pixel_ratio = 0.75
    if mouse_id == 12:
        x_pixel_ratio = 0.75

    xpos_plot = xpos / x_pixel_ratio
    ypos_plot = ypos / y_pixel_ratio

    if mouse_id <= 10:
        a = (point1[1] - point2[1]) / (point1[0] - point2[0])
        b = point1[1] - a * point1[0]
        imedial = np.where(xpos_plot >= -a * ypos_plot + b)[0]
        iv1 = np.where(xpos_plot < -a * ypos_plot + b)[0]
    else:
        # area_names = ['PM','AM','','RL','','LM','AL','','V1','RSP']
        # region_names = ['V1', 'medial', 'anterior', 'lateral', 'all']
        # db = data.db[mouse_id]
        dpath = os.path.join(retinotopy_path, f"{db['mname']}_{db['datexp']}_{db['blk']}.npz")
        aligned_data = np.load(dpath)
        # iregion = aligned_data['iregion']
        iarea = aligned_data['iarea'][ineur]
        imedial = np.where(iarea == 0)[0]
        iv1 = np.where(iarea == 8)[0]
    return iv1, imedial

def load_images(root, file='nat30k.mat', downsample=1, xrange=[0,130], normalize=True, crop=True):
    """ load images from mat file """
    path = os.path.join(root, file)
    dstim = loadmat(path, squeeze_me=True) # stimulus data
    img = np.transpose(dstim['img'], (2,0,1)).astype('float32')
    n_stim, Ly, Lx = img.shape
    print('raw image shape: ', img.shape)

    img = np.array([cv2.resize(im, (int(Lx//downsample), int(Ly//downsample))) for im in img])
    if crop:
        img = img[:,:,:int(176//downsample)] # keep left and middle screen
        xrange = [int(xrange[0]//downsample), int(xrange[1]//downsample)]
        img = img[:,:,xrange[0]:xrange[1]] # crop image based on RF locations
        print('cropped image shape: ', img.shape)
    print('image mean: ', img.mean())
    print('image std: ', img.std())
    if normalize:
        img -= img.mean()
        img /= img.std()
    return img

def load_neurons(file_path, mouse_id = None, fixtrain=False):
    '''
    load neurons of nat60k_text16 recordings.
    file_path: path to the preprocessed file from combine_stim.ipynb file.
    mouse_id: mouse id, used to remove flipped test images for mouse 1,2,3 (2,3 are the l1a2 and l1a3). (optional, since the including of flipped test images doesn't affect the results too much)
    fixtrain: if True, only keep nat30k images for training.
    '''
    print(f'\nloading activities from {file_path}')
    dat = np.load(file_path, allow_pickle=True) 
    # spks_rep = dat['ss'] # (2, 500, NN) two averages of 5 repeats of 500 stim, 10 repeats in total
    # check if ss_all in the file
    spks_rep_all = dat['ss_all'] # 500 x (nrepeats, NN) normally 10 repeats of 500 stim
    ypos, xpos = dat['ypos'], dat['xpos']
    spks = dat['sp']
    istim_sp = (dat['istim_sp']).astype('int')
    istim_ss = (dat['istim_ss']).astype('int')

    if 'nat30k' in file_path: fixtrain = False

    if fixtrain:
        idx = np.where(istim_sp<30000)[0]
        istim_sp = istim_sp[idx]
        spks = spks[:,idx]
    elif mouse_id in [1]: # remove flipped test images for mouse 1
        idx = np.where((istim_sp<30000) | (istim_sp>30500))[0]
        istim_sp = istim_sp[idx]
        spks = spks[:,idx]
    return spks.T, istim_sp, istim_ss, xpos, ypos, spks_rep_all

def split_train_val(istim_train, train_frac=0.9):
    '''
    split training and validation set.
    train_frac: fraction of training set, 1 - train_frac = val_frac.
    '''
    print('\nsplitting training and validation set...')
    print('there is currently no randomness in this function now, please make sure the istim_train is in random order!')
    np.random.seed(0)
    itrain = np.arange(len(istim_train))
    val_interval = int(1/(1-train_frac))
    ival = itrain[::val_interval]
    itrain = np.ones(len(itrain), 'bool')
    itrain[ival] = False
    itrain = np.nonzero(itrain)[0]

    print('itrain: ', itrain.shape)
    print('ival: ', ival.shape)
    return itrain, ival

def normalize_spks(spks, spks_rep, itrain):
    '''
    normalize spks and spks_rep.
    spks: (n_stim_train, n_neurons)
    spks_rep: (n_stim_test, nrepeats, n_neurons)
    itrain: (n_stim_train,)
    '''
    print('\nnormalizing neural data...')
    # spks_mean = spks[itrain].mean(0)
    # spks_std = spks[itrain].std(0)
    # spks_std[spks_std < 0.01] = 0.01
    # spks = spks / spks_std
    # for i in range(len(spks_rep)):
    #     spks_rep[i] /= spks_std
    spks_std = np.nanstd(spks[itrain], axis=0)
    spks_std[(spks_std < 0.01) & (spks_std is not np.nan)] = 0.01
    spks = spks / spks_std
    for i in range(len(spks_rep)):
        spks_rep[i] /= spks_std
    return spks, spks_rep

def nanarray(real_resps,resps):    
    return np.where(real_resps, resps, np.nan)