import numpy as np
import matplotlib.pyplot as plt
import glob
import multiprocessing as mp
from joblib import Parallel, delayed
from astropy.time import Time
import matplotlib.gridspec as gridspec

num_cores = mp.cpu_count()

f = glob.glob('*.plt')
<<<<<<< HEAD
g = glob.glob('*.fil')
=======
>>>>>>> d4a809dcdbb1df453be43047e3d451c166aa0357

def plot(i):
    plt.rcParams['figure.dpi'] = 300
    filename = f[i]
    fname = filename.split('.')
<<<<<<< HEAD
    fname2 = g[0].split("_")
    figname = str(fname[0])

=======
    figname = str(fname[0]) + str('TEST')
    
>>>>>>> d4a809dcdbb1df453be43047e3d451c166aa0357
    x = np.fromfile(filename, dtype='float32')
    mjd = x[0]
    seconds = x[1]
    dm = x[2]
    width = int(x[3])
    x = x[7:]
    x = x.reshape(256,256)
   
    prof = np.sum(x,axis=0)
    maxx = np.max(prof)
    maxi = 0
    maxprof = 0
<<<<<<< HEAD
    #print(seconds)
=======
    print(seconds)
>>>>>>> d4a809dcdbb1df453be43047e3d451c166aa0357
    for i in range(0,255):
        if (prof[i] > maxprof):
            maxi = i
            maxprof = prof[i]
    
    seconds = seconds + (maxi*width)/1000
<<<<<<< HEAD
    #print(seconds)
=======
    print(seconds)
>>>>>>> d4a809dcdbb1df453be43047e3d451c166aa0357
    days = seconds/(3600*24)
    mjdtoa = mjd + days
    time = Time(mjdtoa, format='mjd')
    trueTOA = time.iso

 
    fig = plt.figure(figsize = (8,10))
    gs1 = gridspec.GridSpec(5, 4)
    gs1.update(wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs1[0:-1,0:])
    ax2 = plt.subplot(gs1[-1,0:])

    ax1.grid(False)
    ax2.set_xticks([0,63,127,191,255])
    ax1.set_ylabel('Frequency (MHz)')
    ax2.set_xlabel('time(ms)')
    #
    ax1.set_yticklabels([800,700,600,500,400])
    ax1.set_xticklabels([" "," "," "," "," "])
    ax2.set_xticklabels([0,str(64*width),str(128*width),str(192*width),str(255*width)])
    ax2.set_xlim(-2,258)
   
    im = ax1.imshow(x)
    ax2.plot(prof)

    ax1.set_title(str('MJD: ')+str(int(mjd))+str('   location= ')+str(seconds)+str('s    width=')+str(width)+str('ms/bin    TOA:')+str(trueTOA),fontsize=16)
    plt.savefig(figname,bbox_inches='tight', transparent=True,pad_inches=0)
    plt.close(fig)
    
out = Parallel(n_jobs=num_cores)(delayed(plot)(i) for i in range(len(f)))
