import numpy as np
from skimage.io import imread, imsave

# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxWx3.
#

# Implementation in Fourier Domain / Frankot-Chellappa

def kernpad(K,size):
    ksize = np.asarray(K.shape)
    padsize = np.asarray(size)-ksize
    t = -(ksize-1)/2
    tfin = tuple(t.astype(int))
    Ko1 = np.lib.pad(K,((0,padsize[0]),(0,padsize[1])),'constant')
    Ko = np.roll(Ko1,tfin,axis=(0,1))  
    return Ko

def ntod(nrm, mask, lmda):
    nrm_mat = nrm**mask[:,:,np.newaxis]
    gx = -np.divide(nrm_mat[:,:,0],nrm_mat[:,:,2])
    gx[np.where(mask==0)]=0
    gy = -np.divide(nrm_mat[:,:,1],nrm_mat[:,:,2])
    gy[np.where(mask==0)]=0
    Gx = np.fft.fft2(gx)
    Gy = np.fft.fft2(gy)
    fx = np.array([0.5,0,-0.5]) 
    Fx = kernpad(np.reshape(fx,(1,3)),Gx.shape)
    Fx = np.fft.fft2(kernpad(np.reshape(fx,(1,3)),Gx.shape))
    fy = (-fx.T)
    Fy = np.fft.fft2(kernpad(np.reshape(fy,(3,1)),Gy.shape))
    fr = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])/9
    Fr = np.fft.fft2(kernpad(fr,Gx.shape))
    num = np.conjugate(Fx)*Gx+np.conjugate(Fy)*Gy
    den = np.square(np.absolute(Fx))+np.square(np.absolute(Fy))+lmda*np.square(np.absolute(Fr))+1e-12
    FZ = np.divide(num,den)
    Z = np.real(np.fft.ifft2(FZ))
    return Z


########################## Support code 

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#### Main function

nrm = imread(fn('inputs/phstereo/true_normals.png'))

# Un-comment  next line to read your output instead
# nrm = imread(fn('outputs/prob3_nrm.png'))


mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = np.float32(nrm/255.0)
nrm = nrm*2.0-1.0
nrm = nrm * mask[:,:,np.newaxis]


# Main Call
Z = ntod(nrm,mask,1e-6)


# Plot 3D shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.float32(range(nrm.shape[1])),np.float32(range(nrm.shape[0])))
x = x - np.mean(x[:])
y = y - np.mean(y[:])

Zmsk = Z.copy()
Zmsk[mask == 0] = np.nan
Zmsk = Zmsk - np.nanmedian(Zmsk[:])

lim = 100
ax.plot_surface(x,-y,Zmsk, \
                linewidth=0,cmap=cm.inferno,shade=True,\
                vmin=-lim,vmax=lim)

ax.set_xlim3d(-450,450)
ax.set_ylim3d(-450,450)
ax.set_zlim3d(-450,450)

plt.show()
