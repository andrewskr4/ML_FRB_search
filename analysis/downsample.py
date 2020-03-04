import matplotlib.gridspec as gridspec
import numpy as np
import sys
import matplotlib.pyplot as plt
def plot_sample(filename,figname, fmin, fmax):
    plt.rcParams['figure.dpi'] = 300
    x = np.fromfile(filename, dtype='float32')
    mjd = int(x[0])
    seconds = int(x[1])
    dm = x[2]
    width = int(x[3])
    x = x[7:]
    x = x.reshape(256,256)
    tfmax = 1200-fmin 
    tfmin = 1200-fmax
    
    fdiff = fmax-fmin
    df = 400/256
    downx = [[0 for p in range(256)] for q in range(int(fdiff/df))]
    k=0

    print(int(fdiff/df))
    print(int((fmin-400)/df))
    print(int((fmax-400)/df))

    for i in range(int((tfmin-400)/df), int((tfmax-400)/df)):
        print(i)
        for j in range(0,255):
            #print(j)
            downx[k][j]=x[i][j]
        k+=1

    #print(x128)
    fig = plt.figure(figsize = (8,10))
    gs1 = gridspec.GridSpec(5, 4)
    gs1.update(wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs1[0:-1,0:])
    ax2 = plt.subplot(gs1[-1,0:]) 

    ax1.grid(False)
    ax1.set_xticks([0,63,127,191,255])
    ax1.set_yticks([0,(fdiff/4)-1,(2*fdiff/4)-1,(3*fdiff/4)-1,fdiff-1])
    ax2.set_xticks([0,63,127,191,255])
    #ax2.set_yticks([0,63,127,191,255])
    ax1.set_ylabel('Frequency (MHz)')
    ax2.set_xlabel('time(ms)')
    # ... and label them with the respective list entries
    ax1.set_yticklabels([fmax, 3*fdiff/4 + fmin, 2*fdiff/4 + fmin, fdiff/4 + fmin, fmin])
    ax1.set_xticklabels([" "," "," "," "," "])
    ax2.set_xticklabels([0, str(64*width),str(128*width),str(192*width),str(255*width)])
    ax2.set_xlim(-2,258)
    im = ax1.imshow(downx)


    prof = np.sum(downx,axis=0)
    ax2.plot(prof)
    ax1.set_title(str('MJD: ')+str(int(mjd))+str('   location= ')+str(seconds)+str('s    width=')+str(width),fontsize=8)
    plt.savefig(figname,bbox_inches='tight', transparent=True,pad_inches=0)
    plt.close(fig)


filename = sys.argv[1]
fname = filename.split(".")
outname = fname[0]
#first arguement is name of .dat file, second is the name of the output .png file
plot_sample(str(sys.argv[1]),str(sys.argv[2])+str('-')+str(sys.argv[3])+str('_')+str(outname), int(sys.argv[2]),int(sys.argv[3]))

