import matplotlib.gridspec as gridspec
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

def compare_times(filename, t):
    plt.rcParams['figure.dpi'] = 300
    x = np.fromfile(filename, dtype='float32')
    mjd = int(x[0])
    seconds = int(x[1])
    dm = x[2]
    fnameComp = filename.split(".")
    pngName = fnameComp[0] + '.png'
    width = int(x[3])
    time = int(t)
    if (seconds <= time):
        print("File: "+filename)
        #print("DM: "+str(dm))
        #print("MJD: "+str(mjd))
        #print("Start of Sample (s): "+str(seconds))
        command = 'cp '+filename+' /data/repeaters/chime_frb/prebursts'
        command2 = 'cp '+pngName+' /data/repeaters/chime_frb/prebursts'
        os.system(command)
        os.system(command2)
        #print("Width: "+str(width))



filename = sys.argv[1]
fname = filename.split(".")
outname = fname[0]
#first arguement is name of .dat file, second is the name of the output .png file
compare_times(str(sys.argv[1]), sys.argv[2])
