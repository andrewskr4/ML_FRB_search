import numpy as np
import matplotlib.pylab as plt

import sys


plt.rcParams['figure.dpi'] = 300
x = np.fromfile(sys.argv[1], dtype='float32')
mjd = int(x[0])
seconds = int(x[1])
dm = x[2]
width = int(x[3])
x = x[7:]
x = x.reshape(256,256)
fig, ax = plt.subplots(1,1)
ax.grid(False)
ax.set_xticks([0,64,128,192,256])
ax.set_yticks([0,64,128,192,256])
plt.ylabel('Frequency (MHz)')
plt.xlabel('time(ms)')
# ... and label them with the respective list entries
ax.set_yticklabels([800,700,600,500,400])
ax.set_xticklabels([0,str(64*width),str(128*width),256*width])

im = ax.imshow(x)
plt.title(str('MJD: ')+str(int(mjd))+str('   location= ')+str(seconds)+str('s    width=')+str(width),fontsize=8)
plt.savefig(sys.argv[1]+'.png',bbox_inches='tight', transparent=True,pad_inches=0)
plt.close(fig)

