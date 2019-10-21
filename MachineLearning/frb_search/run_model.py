import torch
import torchvision
import torchvision.transforms as transforms
import sys
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os

@dataclass(eq=False)
class singlePulseSet(Dataset):
    """ Dataset class sublcassed from torch's dataset utility
    Args:
            input_size (int): input size of a datapoint or in this case, the binary size
            start (int): whole number from where the dataset starts counting, exclusive
            end (int): whole number till where the dataset keeps counting, inclusive
    """
    location: str
    
    def __getitem__(self, idx):
        value=idx
        filename = str(self.location)+str('/sample_')+str(value)+str('.plt')
        x = np.fromfile(filename, dtype='float32')
        y = int(x[6])
        x = x[7:]
        x = x-np.mean(x)
        x = x/np.std(x)
        x = np.reshape(x,(256,256))
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        return x, y

    def __len__(self):
        """ setting the length to a limit. Theoretically fizbuz dataset can have
        infinitily long dataset but dataloaders fetches len(dataset) to loop decide
        what's the length. Returning any number from this function sets that as the
        length of the dataset
        """
        l=np.size(glob.glob(str(self.location)+str('/sample_')+str('*')+str('.plt')))
        return l
        
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.drop_out1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(8 * 8 * 512, 4096)
        self.drop_out2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(4096, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
       out = self.layer1(x)
       out = self.layer2(out)
       out = self.layer3(out)
       out = self.layer4(out)
       out = self.layer5(out)
       out = out.reshape(out.size(0), -1)
       out = self.drop_out1(out)
       out = self.fc1(out)
       out = self.drop_out2(out) 
       out = self.fc2(out)
       out = self.fc3(out)
       return out   

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.drop_out1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(8 * 8 * 512, 4096)
        self.drop_out2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(4096, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
       out = self.layer1(x)
       out = self.layer2(out)
       out = self.layer3(out)
       out = self.layer4(out)
       out = self.layer5(out)
       out = out.reshape(out.size(0), -1)
       out = self.drop_out1(out)
       out = self.fc1(out)
       out = self.drop_out2(out) 
       out = self.fc2(out)
       out = self.fc3(out)
       return out             
       

def plot_sample(filename,figname):
    plt.rcParams['figure.dpi'] = 300
    x = np.fromfile(filename, dtype='float32')
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
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('time(ms)')
    # ... and label them with the respective list entries
    ax.set_yticklabels([800,700,600,500,400])
    ax.set_xticklabels([0,str(64*width),str(128*width),256*width])

    im = ax.imshow(x)
    plt.title(str('MJD: ')+str(int(mjd))+str('   location= ')+str(seconds)+str('s    width=')+str(width),fontsize=8)
    plt.savefig(figname,bbox_inches='tight', transparent=True,pad_inches=0)
    plt.close(fig)

      
#trainset = singlePulseSet(0)
testset = singlePulseSet(location=str(sys.argv[1]))
#validationset = singlePulseSet(22500)

#trainloader = DataLoader(trainset, batch_size=10,shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=200,shuffle=False, num_workers=2)
#validationloader = DataLoader(validationset, batch_size=10,shuffle=True, num_workers=2)


model = Net()
#model.load_state_dict(torch.load('/data5/models/best_model.pth'))
model.load_state_dict(torch.load('/data5/models/best_model.pth',map_location='cuda:0'))
model.to(device)

model.eval()

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
correct = 0
total = 0
batch=0
sample=0

num_samples = np.size(glob.glob(str(sys.argv[1])+str('/sample_')+str('*')+str('.plt')))

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device),data[1].to(device)
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        for i in np.arange(200):
            if predicted[i]!=labels[i]:
                x = images[i][0]
                im = plt.imshow(x.cpu().numpy())
                plt.title(str('sample= ')+str(batch*200+i))
                filename = str(sys.argv[1])+str('/sample_')+str(batch*200+i)+str('.plt')
                figname = str(sys.argv[1])+str('/')+str(sys.argv[2])+str('_')+str(sys.argv[3])+str('_sample_')+str(batch*200+i)+str('.png')
                outfile = str(sys.argv[1])+str('/')+str(sys.argv[2])+str('_')+str(sys.argv[3])+str('_sample_')+str(batch*200+i)+str('.dat')
                command = str('mv ')+str(filename)+str(' ')+str(outfile)
                plot_sample(filename,figname)
                os.system(command)
                #plt.savefig(str(sys.argv[1])+str('/sample1_')+str(batch*200+i)+str('.png'),bbox_inches='tight', transparent=True,pad_inches=0)
								#plt.imsave(str(sys.argv[1])+str('/sample_')+str(batch*200+i)+str('.png'),x.cpu().numpy())
                print(str(figname)+str(' saved'))
								#plt.close()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch=batch+1
        print(num_samples,batch*200)
if total==0:
    total=1
print('Accuracy of the network: %2.3f %%' % (100 * correct / total))
