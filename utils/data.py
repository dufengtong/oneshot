import os
import cv2
import numpy as np
from scipy.io import loadmat
from itertools import combinations
import pandas as pd
from scipy.stats import zscore


db = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
# nat30k datasets, due to the high presentation rate, we didn't find similar invariance index results as in Miguel's experiment. So we decide to use his nat15k recording instead for modeling
# db[0]['mname'], db[0]['datexp'], db[0]['blk'] = 'L1_A5', '2023_02_27', '1' # tlag: [3.5, 4.5] nplanes = 2
# db[1]['mname'], db[1]['datexp'], db[1]['blk'] = 'L1_A1', '2023_03_06', '1'# tlag: [3.5, 4.5] nplanes = 2
# db[2]['mname'], db[2]['datexp'], db[2]['blk'] = 'FX9', '2023_05_15', '2' # [4, 5] nplanes = 4
# db[3]['mname'], db[3]['datexp'], db[3]['blk'] = 'FX10', '2023_05_16', '1' # tlags = [4, 5] nplanes = 4
# db[4]['mname'], db[4]['datexp'], db[4]['blk'] = 'FX8', '2023_05_16', '2' # tlags = [4, 5] nplanes = 4
# db[5]['mname'], db[5]['datexp'], db[5]['blk'] = 'FX20', '2023_09_29', '1' # tlags = [4.5, 5.5] nplanes = 2
# db[6]['mname'], db[6]['datexp'], db[6]['blk'] = 'FX40', '2024_10_29', '1' # tlag: [4, 5] nplanes = 4 # no timeline?
# db[7]['mname'], db[7]['datexp'], db[7]['blk'] = 'FX41', '2025_01_23', '1' # tlag: [4, 5] nplanes = 4
# db[8]['mname'], db[8]['datexp'], db[8]['blk'] = 'FX43', '2025_01_24', '1' # tlag: [4, 5] nplanes = 4
# db[9]['mname'], db[9]['datexp'], db[9]['blk'] = 'FX42', '2025_01_25', '1' # tlag: [4, 5] nplanes = 4 -> not used but not sure why
# db[10]['mname'], db[10]['datexp'], db[10]['blk'] = 'FX41', '2025_02_14', '2' # tlag: [4, 5] nplanes = 4
# db[11]['mname'], db[11]['datexp'], db[11]['blk'] = 'FX41', '2025_05_12', '3' # tlag: [4, 5] nplanes = 4
# db[12]['mname'], db[12]['datexp'], db[12]['blk'] = 'FX43', '2025_05_19', '4' # tlag: [4, 5] nplanes = 4
# mouse_names = ['L1_A5', 'L1_A1',  'FX9', 'FX10', 'FX8', 'FX20', 'FX40', 'FX41', 'FX43', 'FX42', 'FX41', 'FX41', 'FX43']
# exp_date = ['022723', '030623', '051523', '051623', '051623', '092923', '102924', '012325', '012425', '012525', '021425', '051225', '051925']
# NNs = [6636,6055,3575,4792,5804,
#        2746, 4261,0,0,6049,
#        5247,3491, 4180]
# NNs_valid = [4242,2840,926,3040,2217,
#              1239, 0,2068,1655,0,
#              886,306,1681]
# img_file_name = ['nat60k_text16_old.mat', # nat60k images and text16 images are original images
#                  'nat60k_text16_old.mat',
#                  'nat60k_text16.mat',# nat60k images are original images, text16 images are rezscored to match nat60k mean and std
#                  'nat60k_text16.mat',
#                  'nat60k_text16.mat',
#                  'nat60k_text16.mat',
#                  '8x4_nat30k_text16.mat',
#                  '8x4_nat30k_text16.mat',
#                  '8x4_nat30k_text16.mat',
#                  '8x4_nat30k_text16.mat',
#                  '8x4_nat30k_text16.mat',
#                  '8x4_nat30k_text16.mat',
#                  '8x4_nat30k_text16.mat',] # medial area recording

# nat15k datasets
# db_g6.append({'mname': 'TX104', 'datexp': '2023_04_06', 'blk':'2', 'stim':'nat15k'})
# db_g6.append({'mname': 'TX110', 'datexp': '2023_04_10', 'blk':'1', 'stim':'nat15k'})
# db_g6.append({'mname': 'TX80', 'datexp': '2022_06_17', 'blk':'1', 'stim':'nat15k'})
# db_g6.append({'mname': 'TX91', 'datexp': '2022_07_28', 'blk':'2', 'stim':'nat15k'})
# db_g6.append({'mname': 'TX115', 'datexp': '2024_01_08', 'blk':'2','stim':'nat15k'})
# db_g6.append({'mname': 'TX114', 'datexp': '2024_01_08', 'blk':'1','stim':'nat15k'})
db[0]['mname'], db[0]['datexp'], db[0]['blk'], db[0]['stim'] = 'TX104', '2023_04_06', '2', 'miguel_passive_15k_8x4.mat'
db[1]['mname'], db[1]['datexp'], db[1]['blk'], db[1]['stim'] = 'TX110', '2023_04_10', '1', 'miguel_passive_15k_8x4.mat'
db[2]['mname'], db[2]['datexp'], db[2]['blk'], db[2]['stim'] = 'TX80', '2022_06_17', '1', 'miguel_passive_15k_8x4.mat'
db[3]['mname'], db[3]['datexp'], db[3]['blk'], db[3]['stim'] = 'TX91', '2022_07_28', '2', 'miguel_passive_15k_8x4.mat'
db[4]['mname'], db[4]['datexp'], db[4]['blk'], db[4]['stim'] = 'TX115', '2024_01_08', '2', 'miguel_passive_15k_8x4.mat'
db[5]['mname'], db[5]['datexp'], db[5]['blk'], db[5]['stim'] = 'TX114', '2024_01_08', '1', 'miguel_passive_15k_8x4.mat'

def zscore_nan(x, axis=1):
    m = np.nanmean(x, axis=axis)
    s = np.nanstd(x, axis=axis)
    # reshape m to match the dimensions of x
    return (x - m[:, None]) / s[:, None]


def get_stim_response_matrix_areas(MouseObject: object, area: int, plane: int, cc_tsh: int = 0.1):
    """
    This function returns a matrix of zscored responses for the given area and plane.

    Parameters
    ----------
    MouseObject : object
        Mouse object containing all the data
    area : int
        Area of the brain to consider
    plane : int
        Plane to consider
    cc_tsh : int
        Threshold for the signal variance coefficient

    Returns
    -------
    zs_rm : np.array
        Matrix of zscored responses for the given area and plane with cc>cc_tsh
        shape (neurons, #stimuli/cats, #reps)
    """

    firstn_cats = 8
    n_instances = 4
    ix_area = np.isin(MouseObject.iarea, area)
    if plane == 1:
        ix_plane = (MouseObject._iplane >= 10)
    elif plane == 2:
        ix_plane = (MouseObject._iplane < 10)
    elif plane == 0:
        ix_plane = np.ones_like(MouseObject._iplane, dtype=bool)
    else:
        raise ValueError("Layer must be 1 or 2 for depths, or 0 for all planes")

    category_id = build_categories(MouseObject.subset_stim) # builds the vector of categories with based on the stimids shape (stimids,)
    stim_ids = MouseObject.subset_stim[category_id <= firstn_cats] # gets the stimids for the first 8 categories (the 32 textures of the 8x4 dataset)
    neurons = MouseObject.neurons_atframes[:,category_id <= firstn_cats] # gets the neurons at frames for the first 8 categories
    cc = sig_variance(neurons, stim_ids) # gets the signal variance for each neuron only for the 8x4 dataset
    if MouseObject.name.startswith(("DR","TX")):
        neurons = zscore(neurons, axis = 1) # for old mice we presented other textures too.
    #neurons_plane_region = MouseObject.neurons_atframes[ix_plane * ix_area] 
    neurons_plane_region = neurons[ix_plane * ix_area]
    cc_plane_region = cc[ix_plane * ix_area]
    if cc_tsh<1: 
        sig_neurons = neurons_plane_region[cc_plane_region>cc_tsh]
    else:
        tsh = np.percentile(cc_plane_region, cc_tsh)
        sig_neurons = neurons_plane_region[cc_plane_region>tsh]
    #print(f"neurons in {area} and plane {plane} with cc>{cc_tsh} : {sig_neurons.shape[0]}")
    total_samples = firstn_cats * n_instances
    _, nc = np.unique(MouseObject.subset_stim, return_counts = True)
    nc = nc[:total_samples]
    nreps = np.min(nc)
    NN = sig_neurons.shape[0]
    stim_response = np.zeros((NN,total_samples,nreps))
    np.random.seed(333)
    for i in range(1,33): # this loop creates the NN, 32, reps matrix
        #instance_idx = np.where(MouseObject.subset_stim == i)[0]
        instance_idx = np.where(stim_ids == i)[0]
        instance_idx = np.random.permutation(instance_idx)
        instance_idx = instance_idx[:nreps]
        stim_response[:,i-1,:] = sig_neurons[:,instance_idx]
    zs_rm = stim_response # fot the FX mice, the 8x4 dataset was the only one used and spks (neurons at frames) come already zscored.
    #sanity check:
    if len(np.where(np.isnan(zs_rm))[0])>0:
        #print("There are NaNs in the zscored representation matrix")
        bad_neurons = np.unique(np.where(np.isnan(zs_rm))[0])
        #print(f"Neuron no.: {bad_neurons}")
        #print(f"Instances: {np.unique(np.where(np.isnan(zs_rm))[1])}")
        #print(f"reps: {np.unique(np.where(np.isnan(zs_rm))[2])}")
        good_neurons = np.array([i for i in range(NN) if i not in bad_neurons])
        #print(f"Keeping {good_neurons.shape[0]} neurons")
        zs_rm = zs_rm[good_neurons,:,:] # only keep the neurons that are not nan 
    return zs_rm

def p_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'
    
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
        point1 = [-600, 0]
        point2 = [-400, 400]
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
        point1 = [-600, 0]
        point2 = [-250, 600]
    if mouse_id == 12:
        point1 = [-800, 0]
        point2 = [-250, 800]
        x_pixel_ratio = 0.75
    # if mouse_id == 11:
    #     hmax_angle = 88
    #     # x_pixel_ratio = 0.75
    # if mouse_id == 12:
    #     x_pixel_ratio = 0.75

    xpos_plot = xpos / x_pixel_ratio
    ypos_plot = ypos / y_pixel_ratio

    if mouse_id <= 12:
        a = (point1[1] - point2[1]) / (point1[0] - point2[0])
        b = point1[1] - a * point1[0]
        imedial = np.where(xpos_plot >= -a * ypos_plot + b)[0]
        iv1 = np.where(xpos_plot < -a * ypos_plot + b)[0]
    else:
        # area_names = ['PM','AM','','RL','','LM','AL','','V1','RSP']
        # region_names = ['V1', 'medial', 'anterior', 'lateral', 'all']
        db_tmp = db[mouse_id]
        dpath = os.path.join(retinotopy_path, f"{db_tmp['mname']}_{db_tmp['datexp']}_{db_tmp['blk']}.npz")
        aligned_data = np.load(dpath)
        # iregion = aligned_data['iregion']
        iarea = aligned_data['iarea'][ineur]
        imedial = np.where(iarea == 0)[0]
        iv1 = np.where(iarea == 8)[0]
    return iv1, imedial

def load_images(root, file='nat30k.mat', downsample=1, xrange=[0,130], normalize=True, crop=True, return_stats=False):
    """ load images from mat file """
    path = os.path.join(root, file)
    dstim = loadmat(path, squeeze_me=True) # stimulus data
    img = np.transpose(dstim['img'], (2,0,1)).astype('float32')
    
    print('raw image shape: ', img.shape)

    # check if the shape is 66x264, if not, resize the image to 66x264 (assert the aspect ratio is the same)
    if img.shape[1] != 66 or img.shape[2] != 264:
        assert img.shape[1]/img.shape[2] == 66/264, 'the aspect ratio of the image is not the same as 66x264, please check the image shape and the downsample factor'
        # resize the image to 66x264
        img = np.array([cv2.resize(im, (264, 66)) for im in img])
        print('resized image shape: ', img.shape)
    n_stim, Ly, Lx = img.shape

    img = np.array([cv2.resize(im, (int(Lx//downsample), int(Ly//downsample))) for im in img])
    if crop:
        img = img[:,:,:int(176//downsample)] # keep left and middle screen
        xrange = [int(xrange[0]//downsample), int(xrange[1]//downsample)]
        img = img[:,:,xrange[0]:xrange[1]] # crop image based on RF locations
        print('cropped image shape: ', img.shape)
    img_mean = img.mean()
    img_std = img.std()
    print('image mean: ', img.mean())
    print('image std: ', img.std())
    if normalize:
        img -= img.mean()
        img /= img.std()
    if return_stats:
        return img, img_mean, img_std
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
    iplane = dat['iplane'] # plane index, 0 for plane 1, 1 for plane 2, etc.

    if 'nat30k' in file_path: fixtrain = False

    if fixtrain:
        idx = np.where(istim_sp<30000)[0]
        istim_sp = istim_sp[idx]
        spks = spks[:,idx]
    elif mouse_id in [1]: # remove flipped test images for mouse 1
        idx = np.where((istim_sp<30000) | (istim_sp>30500))[0]
        istim_sp = istim_sp[idx]
        spks = spks[:,idx]
    return spks.T, istim_sp, istim_ss, xpos, ypos, spks_rep_all, iplane

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

def condense_matrix(ss_m, categories: int = 8, instances: int = 4):
    """
    Condenses a representation matrix into a matrix of mean intra-inter category invariance

    Parameters
    ----------
    ss_m : array
        Representation matrix
    categories : int
        Number of categories
    instances : int
        Number of instances per category

    Returns
    -------
    condensed_matrix : array
        Matrix of mean intra-inter category invariance
        shape (categories, categories)
    """

    total_instances = categories * instances
    cat_init = np.arange(0,total_instances,instances)
    cat_end = np.arange(instances,total_instances+1,instances)
    cats = np.arange(0,categories)
    condensed_matrix = np.zeros((categories,categories))
    for slctd_ct in cats:
        row_from_matrix = ss_m[cat_init[slctd_ct]:cat_end[slctd_ct],:]
        mean_per_row = []
        for cat in cats:
            if cat == slctd_ct:
                cat_responses = row_from_matrix[:,cat_init[cat]:cat_end[cat]]
                a,b = np.triu_indices(instances,1)
                mean = []
                for idx in zip(a,b):
                    mean.append(cat_responses[idx[0],idx[1]])
                mean = np.array(mean).mean()
                mean_per_row.append(mean)
            else:
                inter_mean = row_from_matrix[:,cat_init[cat]:cat_end[cat]].mean()
                mean_per_row.append(inter_mean)
        mean_per_row = np.array(mean_per_row)
        condensed_matrix[slctd_ct,:] = mean_per_row
    return condensed_matrix


def compute_invariance_df(spks, labels, mouse_id, n_categories=8, n_per_category=4):
    """
    spks: (NN, Ntrials) array; labels: (Ntrials,) with 32 unique instances
    grouped into n_categories super-categories of n_per_category each.
    """
    

    spks = zscore(spks, axis=1)

    instances, nc = np.unique(labels, return_counts=True)
    nreps = np.min(nc)
    NN = spks.shape[0]
    n_inst = len(instances)

    stim_response = np.zeros((NN, n_inst, nreps))
    for i, inst in enumerate(instances):
        idx = np.where(labels == inst)[0]
        idx = np.random.permutation(idx)[:nreps]
        stim_response[:, i, :] = spks[:, idx]

    first_h = nreps // 2
    rep_mat = np.zeros((n_inst, n_inst))
    for i in range(n_inst):
        fh = stim_response[:, i, :first_h].mean(axis=1)
        sh = stim_response[:, i, first_h:].mean(axis=1)
        rep_mat[i, i] = np.corrcoef(fh, sh)[0, 1]
    for i, j in combinations(range(n_inst), 2):
        a = stim_response[:, i, :].mean(axis=1)
        b = stim_response[:, j, :].mean(axis=1)
        c = np.corrcoef(a, b)[0, 1]
        rep_mat[i, j] = rep_mat[j, i] = c

    cat = ['Leaves', 'Circles', 'Dryland', 'Rocks',
           'Tiles', 'Squares', 'Rleaves', 'Paved']
    condensed = condense_matrix(rep_mat, n_categories, n_per_category)
    a, b = np.triu_indices(n_categories, 1)
    pos, neg, inv = [], [], []
    for i, j in zip(a, b):
        pos.append(cat[i])
        neg.append(cat[j])
        inv.append(np.mean([condensed[i, i], condensed[j, j]]) - condensed[i, j])

    return pd.DataFrame({
        "positive_category": pos,
        "negative_category": neg,
        "pair_invariance": inv,
        "mouse": mouse_id,
    }), rep_mat



def get_pair_invariance_df(mtx: np.array):
    cat = ['Leaves', 'Circles', 'Dryland', 'Rocks', 'Tiles', 'Squares', 'Rleaves', 'Paved'] 
    areas = ['V1', 'medial', 'lateral', 'anterior']
    n_features = len(mtx.shape)
    layers = [1, 2] 
    df_pair = pd.DataFrame()
    if n_features == 5:
        for m in range(mtx.shape[0]):
            for i_a, area in enumerate(areas):
                for layer in layers:
                    df = pd.DataFrame()
                    matrix = mtx[m, i_a, layer-1, :, :]
                    condensed = condense_matrix(matrix, 8, 4)
                    a, b = np.triu_indices(8,1)
                    positive_cat = []
                    negative_cat = []
                    invariance_index = []
                    for i in zip(a,b):
                        positive_cat.append(cat[i[0]])
                        negative_cat.append(cat[i[1]])
                        invariance_index.append(np.mean([condensed[i[0],i[0]], condensed[i[1],i[1]]]) - condensed[i[0],i[1]])
                    df["positive_category"] = np.array(positive_cat)
                    df["negative_category"] = np.array(negative_cat)
                    df["pair_invariance"] = np.array(invariance_index)
                    df["area"] = area
                    df["layer"] = layer
                    df["mouse"] = m  
                    df_pair = pd.concat([df_pair, df])
        df_pair.reset_index(inplace=True, drop=True)
    elif n_features == 4:
        for m in range(mtx.shape[0]):
            for i_a, area in enumerate(areas):
                df = pd.DataFrame()
                matrix = mtx[m, i_a, :, :]
                condensed = condense_matrix(matrix, 8, 4)
                a, b = np.triu_indices(8,1)
                positive_cat = []
                negative_cat = []
                invariance_index = []
                for i in zip(a,b):
                    positive_cat.append(cat[i[0]])
                    negative_cat.append(cat[i[1]])
                    invariance_index.append(np.mean([condensed[i[0],i[0]], condensed[i[1],i[1]]]) - condensed[i[0],i[1]])
                df["positive_category"] = np.array(positive_cat)
                df["negative_category"] = np.array(negative_cat)
                df["pair_invariance"] = np.array(invariance_index)
                df["area"] = area
                df["mouse"] = m
                df_pair = pd.concat([df_pair, df])
    else:
        raise ValueError("The input matrix must have 4 or 5 dimensions, got %d dimensions." % n_features)
    return df_pair

def compute_pair_inv_model(rep_mtx, categories:int = 8, instances:int = 4):
    cat = ['Leaves', 'Circles', 'Dryland', 'Rocks', 'Tiles', 'Squares', 'Rleaves', 'Paved'] 
    nlayers = rep_mtx.shape[0]
    alex_df_pair = pd.DataFrame()
    for il in range(nlayers):
        df = pd.DataFrame()
        matrix = rep_mtx[il]
        condensed = condense_matrix(matrix, categories, instances)
        a, b = np.triu_indices(categories,1)
        positive_cat = []
        negative_cat = []
        invariance_index = []
        for i in zip(a,b):
            positive_cat.append(cat[i[0]])
            negative_cat.append(cat[i[1]])
            invariance_index.append(np.mean([condensed[i[0],i[0]], condensed[i[1],i[1]]]) - condensed[i[0],i[1]])
        df["positive_category"] = np.array(positive_cat)
        df["negative_category"] = np.array(negative_cat)
        df["pair_invariance"] = np.array(invariance_index)
        df["layer"] = il
        alex_df_pair = pd.concat([alex_df_pair, df])
    alex_df_pair.reset_index(inplace=True, drop=True)
    return alex_df_pair

def compute_model_rep_mtx(resp):
    firstn_cats = 8
    n_instances = 4
    total_samples = firstn_cats * n_instances
    nlayers = resp.shape[0]
    representation_matrix = np.zeros((nlayers,total_samples, total_samples))
    for il in range(nlayers):
        features = resp[il]
        for i in range(total_samples):
            for j in range(total_samples):
                representation_matrix[il,i,j] = np.corrcoef(features[i], features[j])[0,1]
    return representation_matrix


def sig_variance(resp, stimid, use_zscore = False):
    # this function computers signal variance based on repeated presentations of the same stimuli
    # if you have more than two repeats, this can take advantage of that (equivalent to forming all pairs of repeats)

    # resp = (neurons, #stimuli)
    # stimid = (#stimuli)

    # example cc = sig_variance(resp, stimid)

    iunq = np.unique(stimid)
    if use_zscore:
        from scipy.stats import zscore 
        R = zscore(resp, 1)
    else:
        R = resp
    NN = resp.shape[0]
    cc = np.zeros(NN,)    
    nsum = 0
    for j in range(len(stimid)):
        iss = stimid==stimid[j]
        if iss.sum()<2:
            continue;
        cc += (R[:, j] * (R[:, iss].sum(1) - R[:, j])) / (iss.sum()-1)
        nsum += 1 
    if nsum<2:
        raise ValueError('Found %d stimuli with at least two repeats. Requires at least 2.'%nsum)
    cc /= nsum
    return cc