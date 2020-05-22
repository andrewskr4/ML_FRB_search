import matplotlib.gridspec as gridspec
from astropy.time import Time
import numpy as np
import sys
import matplotlib.pyplot as plt
def get_TOA(filename, outname):
    plt.rcParams['figure.dpi'] = 300
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

    for i in range(0,255):
        if (prof[i] > maxprof):
            maxi = i
            maxprof = prof[i]

    seconds = seconds + (i*width)/1000
    days = seconds/(3600*24)
    mjdtoa = mjd + days
    time = Time(mjdtoa, format='mjd')
    trueTOA = time.iso


    f = open("TOA.txt", "w+")

    f.write(trueTOA)
    f.close()
     
    print(mjd)
    
    print(trueTOA)
    
    print(dm)
    


filename = sys.argv[1]
fname = filename.split(".")
outname = fname[0]
#first arguement is name of .dat file
get_TOA(str(sys.argv[1]), outname)

