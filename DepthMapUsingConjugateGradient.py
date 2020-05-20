import numpy as np
from scipy.signal import convolve2d as conv2
from skimage.io import imread, imsave


# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxWx3.

# Implementation using conjugate gradient, with a weight = 0 for mask == 0, and proportional
# to n_z^2 elsewhere. See slides.

def ntod(nrm, mask, lmda):
    w = np.square(nrm[:,:,2])
    w[np.where(mask==0)]=0
    gx = -np.divide(nrm[:,:,0],nrm[:,:,2])
    gx[np.where(mask==0)]=0
    gy = -np.divide(nrm[:,:,1],nrm[:,:,2])
    gy[np.where(mask==0)]=0
    fx = np.array([0.5,0,-0.5]) 
    fy = (-fx.T)
    fr = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])/9
    b = conv2(gx*w,np.reshape(fx[::-1],(1,3)),mode='same',boundary='fill')+conv2(gy*w,np.reshape(fy[::-1],(3,1)),mode='same',boundary='fill')
    Z = np.zeros_like(mask) 
    QZ = conv2((conv2(Z,np.reshape(fx,(1,3)),mode='same',boundary='fill')*w),np.reshape(fx[::-1],(1,3)),mode='same',boundary='fill') \
         +conv2((conv2(Z,np.reshape(fy,(3,1)),mode='same',boundary='fill')*w),np.reshape(fy[::-1],(3,1)),mode='same',boundary='fill')\
         +lmda*conv2(conv2(Z,fr,mode='same',boundary='fill'),np.fliplr(np.flipud(fr)),mode='same',boundary='fill')
    r = b-QZ
    P = np.copy(r)
    for iter in range(200):
        QP = conv2((conv2(P,np.reshape(fx,(1,3)),mode='same',boundary='fill')*w),np.reshape(fx[::-1],(1,3)),mode='same',boundary='fill') \
             +conv2((conv2(P,np.reshape(fy,(3,1)),mode='same',boundary='fill')*w),np.reshape(fy[::-1],(3,1)),mode='same',boundary='fill')\
             +lmda*conv2(conv2(P,fr,mode='same',boundary='fill'),np.fliplr(np.flipud(fr)),mode='same',boundary='fill')
        alpha = np.dot(r.flatten(),r.flatten())/np.dot(P.flatten(),QP.flatten())
        Z = Z+alpha*P
        rprev = np.copy(r)
        r = r-alpha*QP
        beta = np.dot(r.flatten(),r.flatten())/np.dot(rprev.flatten(),rprev.flatten())
        P = r+beta*P
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
Z = ntod(nrm,mask,1e-7)


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
