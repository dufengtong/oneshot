import os 
import numpy as np 
from scipy import io
from pathlib import Path
import numbers

def count_plane_dirs(path):
    return sum(
        1 for name in os.listdir(path)
        if name.startswith("plane") and os.path.isdir(os.path.join(path, name))
    )


def create_ops(root):
    dbnpy = np.load(os.path.join(root, 'db.npy'), allow_pickle=True).item()
    settings = np.load(os.path.join(root, 'settings.npy'), allow_pickle=True).item()
    reg_outputs = np.load(os.path.join(root,'reg_outputs.npy'), allow_pickle=True).item()
    detect_outputs = np.load(os.path.join(root, 'detect_outputs.npy'), allow_pickle=True).item()
    ops_corrected = {**dbnpy, **settings, **reg_outputs, **detect_outputs}
    #old_ops_path = os.path.join(root, 'suite2p', 'plane0', 'ops.npy')
    new_ops_path= os.path.join(root, 'ops.npy')
    np.save(new_ops_path, ops_corrected)
    #os.replace(new_ops_path, old_ops_path) #this will overwrite the old ops file

def load_ops_safe(root):
    ops_path = os.path.join(root, 'ops.npy')
    if os.path.isfile(ops_path):
        ops = np.load(ops_path, allow_pickle=True).item()
        #check if the ops file is complete
        if isinstance(ops['dx'], numbers.Number) == False: ## checks if dx is a number , wrong ops keeps the dx and dy as empty lists []
            print('ops.npy file incomplete, replacing with corrected ops.npy file')
            os.remove(ops_path)
            create_ops(root)
            ops = np.load(ops_path, allow_pickle=True).item()
    else:
        print('ops.npy file not found, creating ops.npy file')
        create_ops(root)
        ops = np.load(ops_path, allow_pickle=True).item()
    return ops

def load_spks(db, irev = np.zeros(30), verbose = True, root_dm11 = '/home/marius/dm11/data/PROC', iscell=False):    
    # this function loads and concatenates planes
    # it automatically applies a 160ms timelag to the deconvolved traces
    # it does not deconvolve the data again
    
    mname, datexp, blk = db['mname'], db['datexp'], db['blk']
    root = os.path.join(root_dm11, mname, datexp, blk)
    
    #ops = np.load(os.path.join(root, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item() 

    ops = load_ops_safe(os.path.join(root, 'suite2p', 'plane0'))
    
    spks = np.zeros((0, ops['nframes']), np.float32) 
    stat = np.zeros((0,)) 
    xpos, ypos, iplane, is_cell = np.zeros((0,)) , np.zeros((0,)) , np.zeros((0,)) , np.zeros((0,2))

    #tlags = .25 + np.linspace(.2, -.8, ops['nplanes']//2+1)[:-1]
    #tlags = np.hstack((tlags, tlags)) + .5
    nrecording_rois = len(list(Path(root).joinpath('suite2p').glob('plane*/'))) 
    print('number of recording planes: %d'%nrecording_rois)
    tlags = 10/60 * ops['fs'] * np.ones(nrecording_rois,)

    for n in range(nrecording_rois):

        #ops = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'ops.npy'), allow_pickle=True).item()
        ops = load_ops_safe(os.path.join(root, 'suite2p', 'plane%d'%n))
        stat0 = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'stat.npy'), allow_pickle=True)
        ypos0 = np.array([stat0[n]['med'][0] for n in range(len(stat0))]) 
        xpos0 = np.array([stat0[n]['med'][1] for n in range(len(stat0))]) 
        if verbose:
            print('plane %d, '%n, 'neurons: %d'%len(xpos0))
        if irev[n]>0:
            ypos0 = -ypos0

        #print('dy, dx offsets: ', ops['dy'], ops['dx'])
        ypos0 += ops['dy'] # add the per plane offsets (dy,dx)
        xpos0 += ops['dx'] # add the per plane offsets (dy,dx)
        
        ypos = np.concatenate((ypos, ypos0), axis=0)
        xpos = np.concatenate((xpos, xpos0), axis=0)

        spks0 = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'spks.npy'), allow_pickle=True)
        if iscell:
            is_cell0 = np.load(
                os.path.join(root, "suite2p", "plane%d" % n, "iscell.npy"),
                allow_pickle=True,
            )
            is_cell = np.concatenate((is_cell, is_cell0), axis=0)
        if tlags[n]<0:
            spks0[:, 1:] = (1 + tlags[n]) * spks0[:, 1:] + (- tlags[n]) * spks0[:, :-1]
        else:
            spks0[:, :-1] = (1 - tlags[n]) * spks0[:, :-1] + tlags[n] * spks0[:, 1:]

        spks0 = spks0.astype('float32')
        iplane = np.concatenate((iplane, n*np.ones(len(stat0),)))
        stat = np.concatenate((stat,stat0), axis=0)     
        if spks.shape[1]>spks0.shape[0]:
            spks0 = np.concatenate((spks0, np.zeros((spks0.shape[0], spks.shape[1]-spks0.shape[1]), 'float32')), axis=1)
        spks = np.concatenate((spks,spks0), axis=0) 
                           

    print('total neurons %d'%len(spks))

    return spks, iplane, xpos, ypos, ops, is_cell

def load_Timeline(db, root_dm11 = '/home/marius/dm11/data/PROC'):    
    # loads Timeline using the loadmat function from scipy.io

    mname, datexp, blk = db['mname'], db['datexp'], db['blk']

    blk = db['blk'] 
    path_suite2p = os.path.join(root_dm11, mname, datexp, blk)
    fname     = 'Timeline_%s_%s_%s'%(mname, datexp, blk) 
    fnamepath = os.path.join(path_suite2p, fname) 
    Timeline  = io.loadmat(fnamepath, squeeze_me=True)['Timeline'] 
    
    return Timeline


def stim_binning(Timeline, spks, ops, bid = 0, tlag = 0):
    # bid is the block id in the recording relative to all blocks in ops['frame_per_folder']
    # tlag is a timelag. should be 0 for all recordings (a 160ms delay is already applied in load_spks)
    
    cframe = np.cumsum(ops['frames_per_folder'])
    cframe = np.hstack((np.zeros(1,'int32'), cframe))
    #print(cframe)

    NN, NT = spks.shape
    frame_start = Timeline['stiminfo'].item()['frame_start']
    istim = Timeline['stiminfo'].item()['istim']

    frame_start = np.array(frame_start).astype('int')

    frame_start0 = frame_start + tlag
    ix = frame_start0+10<cframe[bid+1] - cframe[bid]#//2
    frame_start0 = frame_start0[ix]
    sstim = istim[ix]

    S  = spks[:,  frame_start0 + cframe[bid]]
    
    return S, sstim


def signal_variance(sstim, S):
    NN = S.shape[0]
    istim = np.unique(sstim)
    k = 0
    s2 = np.zeros((2, len(istim), NN), 'float32')
    for j in range(len(istim)):
        ix = np.nonzero(sstim==istim[j])[0]
        if len(ix)==2:
            s2[0, k] = S[:, ix[0]]
            s2[1, k] = s2[:, ix[1]]
            k +=1
    s2 = s2[:,:k]


    ss = s2 - np.mean(s2,1)[:,np.newaxis,:]
    ss = ss / np.mean(ss**2,1)[:,np.newaxis,:]**.5 + 1e-10

    csig = (ss[0] * ss[1]).mean(0)
    print('signal variance is %2.2f'%csig.mean())
    if csig.mean()<0.05:
        print('The signal variance should be at least 0.05 for a normal window with most neurons in visual cortex.')

    return csig, ss


def get_neurons_atframes(timeline, spks, bin=None):
    """
    Get the neurons at each frame, and the subset of stimulus before the recording ends.

    Parameters
    ----------
    spks : array
        Spikes of the neurons.
    Timeline : array
        Timeline of the experiment.

    Returns
    -------
    neurons_atframes : array
        Neurons at each frame.
    subset_stim: array
        Stimuli before recording ends
    """
    def rolling_window(a,win_size=bin,freq=bin):
        return np.lib.stride_tricks.sliding_window_view(a,win_size)[::freq,:].mean(1)

    if bin is None:
        _, nt = spks.shape
        tlag = 1  # this is the normal lag between frames and stimuli
        istim = timeline["stiminfo"].item()["istim"]
        frame_start = timeline["stiminfo"].item()["frame_start"]
        frame_start = np.array(frame_start).astype("int")
        frame_start0 = frame_start + tlag
        ix = frame_start0 < nt
        frame_start0 = frame_start0[ix]
        neurons_atframes = spks[
            :, frame_start0
        ]  # sample the neurons at the stimulus frames
        subset_stim = istim[ix]
    else:
        print(f"binning spikes with bin size {bin}")
        _, nt = spks.shape
        tlag = 1  # this is the normal lag between frames and stimuli
        istim = timeline["stiminfo"].item()["istim"]
        frame_start = timeline["stiminfo"].item()["frame_start"]
        frame_start = np.array(frame_start).astype("int")
        frame_start0 = frame_start + tlag
        ix = frame_start0 < nt
        frame_start0 = frame_start0[ix]
        n_stim_frames = len(frame_start0)
        frames_to_bin = np.empty((0,n_stim_frames*bin),int)
        for stim_frame in range(n_stim_frames):
            effective_frames = np.arange(frame_start0[stim_frame],frame_start0[stim_frame]+bin)
            frames_to_bin = np.append(frames_to_bin, [effective_frames])
        frames_to_bin = frames_to_bin[frames_to_bin<nt]
        neurons_atframes_tobin = spks[:, frames_to_bin]
        neurons_atframes = np.apply_along_axis(rolling_window, 1, neurons_atframes_tobin)  # bin the neurons using specified bin
        subset_stim = istim[ix][:neurons_atframes.shape[1]]
    return neurons_atframes, subset_stim