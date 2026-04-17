import os
import numpy as np
import h5py
from scipy.ndimage import uniform_filter
from scipy.interpolate import interp1d 
import mat73

def tun_curves(Z, data, dstim, dt = np.arange(6,12), nstim = 200, nplanes = 1):
    dstim_type = dstim[1]
    igood = (dstim_type==0).astype('float32')
    #igood = uniform_filter(igood, 300) <.01
    igood = igood <.01

    iframe = np.logical_and(data[1, :-1] >2.5 , data[1, 1:] <2.5).nonzero()[0]
    iframe = iframe[::nplanes]

    istamp = (data[2, :] > 100).nonzero()[0]
    frame_time = np.interp(iframe, istamp, data[2, istamp])

    dstim_time = dstim[0]
    findex = np.interp(dstim_time, frame_time, np.arange(len(frame_time)), right = -1, left = -1)

    dstim_type = dstim[1]

    NN = Z.shape[0]
    nsp = np.zeros((2, nstim, NN, len(dt)), 'float32')
    strace = np.zeros((nstim, Z.shape[1]), 'bool')

    for istim in range(nstim):
        irep_start = ((dstim_type==istim+1) * igood * (findex>=0)).astype('float32')

        irep_start = irep_start.nonzero()[0]

        first_frame = np.round(findex[irep_start]).astype('int32')
        first_frame = first_frame[first_frame + dt[-1] < Z.shape[1]]

        strace[istim, first_frame] = 1

        iframe = first_frame[:,np.newaxis] + dt
        #import pdb; pdb.set_trace()

        nmax = np.minimum(200, len(iframe))
        nsp[0, istim] = Z[:, iframe[:nmax:2]].mean(-2)#.mean(-1)
        nsp[1, istim] = Z[:, iframe[1:nmax:2]].mean(-2)#.mean(-1)
    return nsp

def load_dataset(db, iplanes = None, deconv = 0, subcells = True, datapath='/home/carsen/dm11_pachitariu/data/PROC'): #, nsub = 1):    
    from suite2p.extraction import dcnv

    mname, datexp, blk = db['mname'], db['datexp'], db['blk']
    root = os.path.join(datapath, mname, datexp, blk)

    ops = np.load(os.path.join(root, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item() # ops is a "pickled" object,

    if iplanes is None:
        iplanes = np.arange(ops['nplanes'])
    # loop over planes and concatenate
    
    for n in iplanes: 
        ops = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'ops.npy'), allow_pickle=True).item()

        stat = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'stat.npy'), allow_pickle=True)        
        
        F0 = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'F.npy'))
        Fneu0 = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'Fneu.npy'))
        
        F0 = F0 - 0.7*Fneu0
        #F0 = Fneu0

        
        if subcells:
            icell = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'iscell.npy'))[:,0]
        else:            
            icell = np.ones((len(F0),), 'bool')

        ypos0 = np.array([stat[n]['med'][0] for n in range(len(stat))])[icell>.5] 
        xpos0 = np.array([stat[n]['med'][1] for n in range(len(stat))])[icell>.5] 
        
        if 'dy' in ops:
            ypos0 += ops['dy'] # add the per plane offsets (dy,dx)
            xpos0 += ops['dx'] # add the per plane offsets (dy,dx)


        F0 = F0[icell>.5]

        #F0 = F0[:, ::nsub]

        Fn = dcnv.preprocess(F0, ops['baseline'], 2 * ops['win_baseline'],
                             ops['sig_baseline'], ops['fs'] , ops['prctile_baseline']) # / nsub

        Fb = F0 - Fn
        #F0 = (F0 - Fb) / np.maximum(10, Fb)
        F0 = (F0 - Fb) / np.maximum(10, Fb.mean())

        if deconv:
            F0     = dcnv.oasis(F0, ops['batch_size'], ops['tau'], ops['fs']) #/ nsub

        if n==0:
            F = F0
            xpos = xpos0
            ypos = ypos0
            iplane = np.zeros_like(xpos0)
        else:
            NN = F.shape[0]
            F = np.concatenate((F,np.zeros((F0.shape[0], F.shape[1]), dtype='float32')), axis=0)
            F[NN:, :F0.shape[1]] = F0

            xpos = np.concatenate((xpos, xpos0), 0)
            ypos = np.concatenate((ypos, ypos0), 0)
            iplane = np.concatenate((iplane, n * np.ones_like(xpos0)), 0)

        print('plane %d, '%n, 'neurons: %d'%F0.shape[0])

    print('total neurons %d'%F.shape[0])
    return F, ops, xpos, ypos, iplane


def stimVR(dat, spks):
    # for fake VR only 
    tdaq = dat['TL']['daq']['time']
    tdaq = np.array(tdaq)
    idaq = 80 * (1+np.arange(len(tdaq)))
    ddaq = dat['TL']['daq']['data'].T.flatten()
    ix = (ddaq[1:] < 2) * (ddaq[:-1]>2)
    imic = ix.nonzero()[0]

    imic = imic[:spks.shape[1]]
    tmic = np.interp(imic, idaq, tdaq)

    istim = dat['TL']['stim']['istim']-1
    tstim = dat['TL']['stim']['time']
    istim = istim[:len(tstim)]

    nsub = 1
    ilen = (istim==istim[0]).nonzero()[0][2] 

    ntrials = len(tstim) // ilen
    tmax = ntrials * ilen
    tstim = tstim[:tmax]
    istim = istim[:tmax]

    isub = np.round(np.linspace(0, ilen, 1+ilen//nsub)[:-1]).astype('int32')

    tstim = np.reshape(tstim, (-1, ilen))  + 10/60 * 1/(24*3600)
    tstim = tstim[:, isub].flatten()

    f = interp1d(tmic, spks[:, :len(tmic)])
    sp = f(tstim)
    NN = spks.shape[0]
    sp = sp.reshape(NN, ntrials, -1)

    return sp

def stim_and_mic_time(dat,  spks, tlags = [4.5, 5.5], nplanes = 1):
    #  5/16/23 changed default tlags from [3.5, 4.5] to [4.5, 5.5]

    # spks is NN by NT
    # tlags at 30 Hz get summed up
    # dat contains the Timeline TL variable

    # this block extracts microscope frame times
    daq_period = dat['TL']['daq']['data'].shape[0]


    tdaq = dat['TL']['daq']['time']
    tdaq = np.array(tdaq)
    idaq = daq_period * (1+np.arange(len(tdaq)))
    ddaq = dat['TL']['daq']['data'].T.flatten()
    ix = (ddaq[1:] < 2) * (ddaq[:-1]>2)
    imic = ix.nonzero()[0]
    imic = imic[:spks.shape[1]*nplanes:nplanes]
    tmic = np.interp(imic, idaq, tdaq)
    # print('tmic', tmic.shape, tmic.min(), tmic.max())

    
    # the stimulus times and indices, 1-indexed
    istim = dat['TL']['stim']['istim'] - 1
    istim = istim.T.flatten()

    tstim = dat['TL']['stim']['time']
    istim = istim[:len(tstim)]#.astype('int32')

    # this interpolates and adds together frames at lag 3.5 and 4.5 from the stimulus
    # print(len(tmic), spks.shape)
    f = interp1d(tmic, spks, fill_value = 'extrapolate')
    sp = f(tstim + tlags[0]/30 / (24 * 3600)) 
    for j in range(1, len(tlags)):
        sp += f(tstim + tlags[j]/30 / (24 * 3600))

    sp = sp.astype('float32') / len(tlags)
    
    return sp, istim

def load_timeline(db, datapath='/home/carsen/dm11_pachitariu/data/PROC'):    
    mname, datexp, blk = db['mname'], db['datexp'], db['blk']
    root = os.path.join(datapath, mname, datexp, blk)
    fname     = 'Timeline_%s_%s_%s_RAW.mat'%(mname, datexp, blk) 
    fnamepath = os.path.join(root, fname) 

    data = mat73.loadmat(fnamepath)

    return data
        

def trial_average(spks, data, dstim, p, dt, reps=False):
    Z = spks.copy()
    #Z = Z - np.mean(Z, axis=1)[:,np.newaxis]

    dstim_type = dstim[1]
    igood = (dstim_type==0).astype('float32')
    igood = igood <.01
    if reps is True:
        ifirst = dstim_type != np.roll(dstim_type, 1)
        igood = np.logical_and(igood, ifirst)

    #igood = uniform_filter(igood, 300) <.01

    nstim = int(np.max(dstim[1]))
    NN = spks.shape[0]
    sfr = np.zeros((2, nstim, NN, len(dt)), 'float32')

    strace = np.zeros((nstim, spks.shape[1]), 'bool')

    for istim in range(nstim):
        iframe = np.logical_and(data[1, :-1] >2.5 , data[1, 1:] <2.5).nonzero()[0]
        istamp = (data[2, :] > 100).nonzero()[0]
        frame_time = np.interp(iframe, istamp, data[2, istamp])

        dstim_time = dstim[0]
        findex = np.interp(dstim_time, frame_time, np.arange(len(frame_time)), right = -1, left = -1)

        dstim_type = dstim[1]
        irep_start = ((dstim_type==istim+1) * igood * (findex>=0)).astype('float32')

        irep_start = irep_start.nonzero()[0]

        first_frame = np.round(findex[irep_start]).astype('int32')
        first_frame = first_frame[first_frame + dt[-1] < Z.shape[1]]

        strace[istim, first_frame] = 1

        iframe = first_frame[:,np.newaxis] + dt

        sfr[0,istim] = Z[:, iframe[::2]].mean(1)
        sfr[1,istim] = Z[:, iframe[1::2]].mean(1)
    return sfr


def find_kernels(sfr, dt, dcnv = False):
    NN = sfr.shape[2]
    kern = np.zeros((NN, len(dt)))
    for t in range(NN):
        FR  = sfr[0, :, t, :]#.mean(0)
        FR1 = sfr[1, :, t, :]
        if dcnv is False:
            rr = FR[:, 27:60].mean(1)
        else:
            rr = FR[:, 27:40].mean(1)
        isort = np.argsort(rr)[::-1]
        kern[t] = np.mean(FR1[isort[:5], :], 0)

    return kern

def quant_kernels(kern, dcnv = False):
    if dcnv is False:
        c = kern[:,27:60].mean(1) /kern[:,:25].mean(1)
    else:
        c = kern[:,27:40].mean(1) /kern[:,:25].mean(1)

    ts = np.linspace(0, 124, 125)
    tups = np.linspace(0, 124, 1241)
    sig = 1

    Kyx = np.exp(-(tups[:, np.newaxis] - ts)**2 / sig**2)
    Kxx = .1 * np.eye(len(ts))  + np.exp(-(ts[:, np.newaxis] - ts)**2 / sig**2)

    K = np.linalg.solve(Kxx, Kyx.T)

    kern0 = kern - kern[:,:25].mean(1)[:, np.newaxis]
    ker = kern0 @ K

    imax = np.argmax(ker, axis=1)
    kmax = np.max(ker, axis=1)

    hdcy = np.zeros((ker.shape[0],))
    for j in range(ker.shape[0]):
        hdcy[j] = np.nonzero(ker[j] > kmax[j]/2)[0][-1] - imax[j]
        ker[j] = np.roll(ker[j], 400 - imax[j])

    return c, hdcy

def signal_variance(sstim, S):
    NN = S.shape[0]
    istim = np.unique(sstim)
    k = 0
    s2 = np.zeros((2, len(istim), NN), 'float32')
    for j in range(len(istim)):
        ix = np.nonzero(sstim==istim[j])[0]
        if len(ix)>=2:
            s2[0, k] = S[:, ix[::2]].mean(-1)
            s2[1, k] = S[:, ix[1::2]].mean(-1)
            k +=1
    s2 = s2[:,:k]


    ss = s2 - np.mean(s2,1)[:,np.newaxis,:]
    ss = ss / np.mean(ss**2,1)[:,np.newaxis,:]**.5 + 1e-10

    csig = (ss[0] * ss[1]).mean(0)
    print('signal variance is %2.2f'%csig.mean())
    if csig.mean()<0.05:
        print('The signal variance should be at least 0.05 for a normal window with most neurons in visual cortex.')

    return csig, ss
