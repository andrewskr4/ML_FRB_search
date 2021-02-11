import matplotlib.gridspec as gridspec
import numpy as np
import sys
import matplotlib.pyplot as plt
from astropy.time import Time


def plot_sample(filename,figname):
    plt.rcParams['figure.dpi'] = 300
    x = np.fromfile(filename, dtype='float32')
    #print(len(x))
    mjd = x[0]
    seconds = int(x[1])
    dm = x[2]
    width = int(x[3])
    x = x[7:]
    x = x.reshape(256,256)
   

    #####Uncomment to add pulse profile (also need to uncomment in plotting area)
    prof = np.sum(x,axis=0)

    ####Uncomment the folling lines if TOA is desired
<<<<<<< HEAD
    seconds = "???"
    trueTOA = "???"
    #maxx = np.max(prof)
    #maxi = 0
    #maxprof = 0
    
    #for i in range(0,255):
    #    if (prof[i] > maxprof):
    #        maxi = i
    #        maxprof = prof[i]
    
    #seconds = seconds + (maxi*width)/1000
=======
    #seconds = "???"
    #trueTOA = "???"
    maxx = np.max(prof)
    maxi = 0
    maxprof = 0
    
    for i in range(0,255):
        if (prof[i] > maxprof):
            maxi = i
            maxprof = prof[i]
    
    seconds = seconds + (maxi*width)/1000
>>>>>>> dfb624cf32fd09605971a5e459d3e001140e3056


    #f = open("Location.txt", "w+")

    #f.write(str(seconds))
    #f.close()
    
<<<<<<< HEAD
    #days = seconds/(3600*24)
    #mjdtoa = mjd + days
    #time = Time(mjdtoa, format='mjd')
    #trueTOA = time.iso
=======
    days = seconds/(3600*24)
    mjdtoa = mjd + days
    time = Time(mjdtoa, format='mjd')
    trueTOA = time.iso
>>>>>>> dfb624cf32fd09605971a5e459d3e001140e3056
    #################################################
 
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
    ####Uncomment to add pulse profile
    ax2.plot(prof)

    ax1.set_title(str('MJD: ')+str(int(mjd))+str('   location= ')+str(seconds)+str('s    width=')+str(width)+str('ms/bin    TOA:')+str(trueTOA),fontsize=16)
    plt.savefig(figname,bbox_inches='tight', transparent=True,pad_inches=0)
    plt.close(fig)


filename = sys.argv[1]
fname = filename.split(".")
outname = fname[0]
#first arguement is name of .dat file, second is the name of the output .png file
#plot_sample(str(sys.argv[1]),str(sys.argv[2])+str(outname)) 
plot_sample(str(sys.argv[1]),str(outname))
print("Done")















