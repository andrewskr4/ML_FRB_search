import matplotlib.gridspec as gridspec
import numpy as np
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

def plot_sample(filename, fmin, fmax):
    plt.rcParams['figure.dpi'] = 300
    x = np.fromfile(filename, dtype='float32')
    mjd = x[0]
    seconds = x[1]
    dm = x[2]
    width = x[3]
    temp1 = x[4]
    temp2 = x[5]
    temp3 = x[6]
    x = x[7:]
    x = x.reshape(256,256)
    tfmax = 1200-fmin
    tfmin = 1200-fmax

    df = 400/256
    for i in range(int((tfmin-400)/df),int((tfmax-400)/df)):
        x[i,:] = 0

    x = x.reshape(65536)
    x = np.float32(x)
    
    fname = filename.split('.')
    fname = str(fname[0]) + str('zapped.dat')
    #f = open(str(fname), 'w+')
    #print(x)
    
    output = np.array([])
    output = np.append(output, np.float32(mjd))
    output = np.append(output, np.float32(seconds))
    output = np.append(output, np.float32(dm))
    output = np.append(output, np.float32(width))
    output = np.append(output, np.float32(temp1))
    output = np.append(output, np.float32(temp2))
    output = np.append(output, np.float32(temp3))
    output = np.concatenate((output, x), axis=None)
    outfile = open("text_test.dat", mode='wb')
    print(len(output))
    output.tofile(outfile)
    outfile.close()

plot_sample(str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

