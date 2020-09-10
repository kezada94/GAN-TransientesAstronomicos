import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

def create_noisy_xor(N_per_cluster=500, stddev_noise=0.4):
    data = stddev_noise*np.random.randn(4*N_per_cluster, 2)
    data[0*N_per_cluster:1*N_per_cluster, :] += [1.0, -1.0]
    data[1*N_per_cluster:2*N_per_cluster, :] += [-1.0, 1.0]
    data[2*N_per_cluster:3*N_per_cluster :] += [-1.0, -1.0]
    data[3*N_per_cluster:4*N_per_cluster, :] += [1.0, 1.0]
    #data = (data - np.mean(X, axis=0))/np.std(X, axis=0)
    labels = np.zeros(shape=(4*N_per_cluster,), dtype=int)
    labels[2*N_per_cluster:] = 1.0
    NP = np.random.permutation(4*N_per_cluster)
    return data[NP, :], labels[NP]

def featurize_lc(lc_data, period, phi_interp, sp=0.15): 
    mjd, mag, err = lc_data.T
    phi = np.mod(mjd, period)/period
    mag_interp = np.zeros_like(phi_interp)
    err_interp = np.zeros_like(phi_interp)
    w = 1.0/err**2
    for i in range(len(phi_interp)):
        gt = np.exp((np.cos(2.0*np.pi*(phi_interp[i] - phi)) -1)/sp**2)
        norm = np.sum(w*gt)
        mag_interp[i] = np.sum(w*gt*mag)/norm
        err_interp[i] = np.sqrt(np.sum(w*gt*(mag - mag_interp[i])**2)/norm)
    err_interp += np.sqrt(np.median(err**2))
    idx_max =  np.argmin(mag_interp)
    mag_interp = np.roll(mag_interp, -idx_max)
    err_interp = np.roll(err_interp, -idx_max)
    max_val = np.amax(mag_interp + err_interp)
    min_val = np.amin(mag_interp - err_interp)
    mag_interp = 2*(mag_interp - min_val)/(max_val - min_val) - 1
    err_interp = 2*err_interp/(max_val - min_val)
    return mag_interp, err_interp, [max_val, min_val, idx_max]

def defeaturize_lc(mag, err, norm):
    # center, scale, idx_max = norm[0], norm[1], norm[2]
    max_val, min_val, idx_max = norm[0], norm[1], norm[2]
    idx_max = int(idx_max)
    return 0.5*(np.roll(mag, idx_max) +1)*(max_val - min_val) + min_val, 0.5*np.roll(err, idx_max)*(max_val - min_val)


class live_metric_plotter:
    """
    This create and update the plots of the reconstruction error  and the KL divergence
    """
    def __init__(self, figsize=(7, 3)):
        self.fig, ax1 = plt.subplots(1, figsize=figsize, tight_layout=True)
        ax2 = ax1.twinx() 
        ax2.set_ylabel('KL qzx||pz (dotted)');
        ax1.set_ylabel('-log pxz (solid)')
        ax1.set_xlabel('Epoch')
        ax1.plot(0, alpha=0.75, linewidth=2, label='Train') 
        ax1.plot(0, alpha=0.75, linewidth=2, label='Validation')
        ax2.plot(0, alpha=0.75, linewidth=2, label='Train', linestyle='--') 
        ax2.plot(0, alpha=0.75, linewidth=2, label='Validation', linestyle='--')
        plt.legend(); plt.grid(); 
        self.axes = list([ax1, ax2])   
        
    def update(self, epoch, metrics):
        for i, ax in enumerate(self.axes):
            for j, line in enumerate(ax.lines):
                line.set_data(range(epoch+1), metrics[:epoch+1, j, i])
            ax.set_xlim([0, epoch])
            ax.set_ylim([np.amin(metrics[:epoch+1, :, i]), np.amax(metrics[:epoch+1, :, i])])
        self.fig.canvas.draw();
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        # z = 100 x 1 x 1
        self.conv1_1 = nn.ConvTranspose2d( nz, ngf * 2, 5, 1, 0, bias=False)
        self.bn1_1 = nn.BatchNorm2d(ngf * 2)
        
        self.conv1_2 = nn.ConvTranspose2d( 2, ngf * 2, 5, 1, 0, bias=False)
        self.bn1_2 = nn.BatchNorm2d(ngf * 2)
        
        # state size. 336 x 5 x 5

        self.conv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 2)
        # state size. 168 x 11 x 11

        self.conv3 = nn.ConvTranspose2d( ngf * 2, ngf, 5, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf)

        # state size. (ngf*2) x 21 x 21

        self.conv4 = nn.ConvTranspose2d( ngf, nc, 6, 2, 2, bias=False)

        # state size. (ngf) x 42 x 42

        self.fin = nn.Tanh()
        self.act = nn.ReLU(True)
        
        # state size. (3) x 42 x 42
        

    def forward(self, input, label):
        
        x = self.act(self.bn1_1(self.conv1_1(input)))
        y = self.act(self.bn1_2(self.conv1_2(label)))
        #print(x.shape)
        #print(y.shape)
        
        x = torch.cat([x, y], 1)
        #print(x.shape)

        x = self.act(self.bn2(self.conv2(x)))
        #print(x.shape)

        x = self.act(self.bn3(self.conv3(x)))
        #print(x.shape)

        x = self.fin(self.conv4(x))
        #print(x.shape)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()       
        
            # input is (nc) x 42 x 42
        self.conv1_1 = nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False)
        self.conv1_2 = nn.Conv2d(2, ndf//2, 4, 2, 1, bias=False)
            # state size. (ndf) x 21 x 21
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
            # state size. (ndf*2) x 10 x 10
            
        
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
            # state size. (ndf*4) x 5 x 5
            
            
        self.conv4 = nn.Conv2d(ndf * 4, 1, 6, 2, 1, bias=False)
        # state size. (ndf*8) x 2 x 2
            
        #self.conv5 = nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False)
        self.capa5 = nn.Linear(ndf*8*2*2, 1)
        self.fin = nn.Sigmoid()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, label):
        #print(x.shape)
        x = self.act(self.conv1_1(x))
        y = self.act(self.conv1_2(label))
        
        x = torch.cat([x, y], 1)
        #print(x.shape)
        
        x = self.act(self.bn1(self.conv2(x)))
        #print(x.shape)
        
        x = self.act(self.bn2(self.conv3(x)))
        #print(x.shape)
        
        #x = self.act(self.bn3(self.conv4(x)))
        x = self.fin(self.conv4(x))
        #x = x.view(x.shape[0], -1)
        #print(x.shape)
        
        #x = self.fin(self.capa5(x))
        
        return x