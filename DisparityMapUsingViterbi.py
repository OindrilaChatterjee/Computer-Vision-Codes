## Default modules imported. Import more if you need to.

import numpy as np


#########################################
### Hamming distance computation
### You can call the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the elmenet-wise hamming distance.
hd8bit = np.zeros((256,))
for i in range(256):
    v = i
    for k in range(8):
        hd8bit[i] = hd8bit[i] + v%2
        v=v//2


def hamdist(x,y):
    dist = np.zeros(x.shape)
    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256
    return dist
#########################################

#
## Copy this from solution to problem 2.
#def buildcv(left,right,dmax):
#    cv = 24 * np.ones([left.shape[0],left.shape[1],dmax+1], dtype=np.float32)
#    return cv

def census(img):
    
    [H,W] = img.shape
    img_pad = np.pad(img,(2,),'constant',constant_values=(255,))
    c = np.zeros([H,W],dtype=np.uint32)
    ind = 0
    for i in range(5):
        for j in range (5):
            if (abs(i)+abs(j)!=0):
                neighbormat = img_pad[i:i+H,j:j+W]
                csum = np.array(1*(img>neighbormat),dtype=np.uint32)
                c = c+pow(2,23-ind)*csum
                ind+=1
    
    return c
#
# You can call the hamdist function above, and copy your census function from the
# previous problem set.
def buildcv(left,right,dmax):
    d = np.zeros(left.shape)
    cv = 24 * np.ones([left.shape[0],left.shape[1],dmax+1], dtype=np.float32)
    cenleft = census(left)
    cenright = census(right)
    [H,W] = left.shape
    for i in range(0,dmax+1):
        hd = hamdist(cenleft[:,i:],cenright[:,0:W-i])
        cv[:,i:,i] = hd
    return cv


# Implement the forward-backward viterbi method to smooth
# only along horizontal lines. Assume smoothness cost of
# 0 if disparities equal, P1 if disparity difference <= 1, P2 otherwise.
#
# Function takes in cost volume cv, and values of P1 and P2
# Return the disparity map
def viterbilr(cv,P1,P2):

    [H,W,D] = cv.shape
    cbar = np.copy(cv)
    cres = np.zeros_like(cv)
    z = np.zeros([H,W,D])
    dvec = np.arange(D)
    d = np.broadcast_to(dvec,[H,D])
 # Forward pass
    for x in range(1,W):
        
        compmat = np.zeros([H,D,4])
        argcompmat = np.zeros([H,D,4])
        compmat[:,:,0] = P2*np.ones([H,D])
        cbarprevcol = cbar[:,x-1,:]
        mincbprevcol = np.min(cbarprevcol,axis=1)
        compmat[:,:,1] = np.roll(cbarprevcol,-1,axis=1)-mincbprevcol[:,np.newaxis]+P1*np.ones([H,D])
        compmat[:,:,3] = np.roll(cbarprevcol,1,axis=1)-mincbprevcol[:,np.newaxis]+P1*np.ones([H,D])
        compmat[:,:,2] = cbarprevcol-mincbprevcol[:,np.newaxis]
        mincomp = np.min(compmat,axis=2)
        cres[:,x,:] = mincomp+mincbprevcol[:,np.newaxis]
        cbar[:,x,:] = cres[:,x,:]+ cv[:,x,:]
        argmincomp = np.argmin(compmat,axis=2)
        
        argminres = np.reshape(np.argmin(cres[:,x,:],axis=1),(H,1))
        argcompmat[:,:,0] = np.broadcast_to(argminres,[H,D])
        argcompmat[:,:,1] = np.roll(d,-1,axis=1)
        argcompmat[:,:,3] = np.roll(d,1,axis=1)
        argcompmat[:,:,2] = np.copy(d) 
        t = np.zeros([H,D])
        maskmat =np.zeros([H,D,4])
        for ind in range(4):
            mask = 1*(argmincomp==ind)
            maskmat[:,:,ind] = mask*argcompmat[:,:,ind]
        z[:,x,:] = np.sum(maskmat,axis=2)
# Backward pass
    dnewmap = np.zeros([H,W])
    dend = np.argmin(cbar[:,-1,:],axis=1)
    dnewmap[:,W-1] = dend
    for xback in range(0,W-1):       
        j = W-2-xback
        prevd = dnewmap[:,j+1].astype(int)
        prevz = z[:,j+1,:]
        t = np.arange(H)
        dnewmap[:,j] = prevz[t,prevd]
    return dnewmap
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = np.float32(imread(fn('inputs/left.jpg')))/255.
right = np.float32(imread(fn('inputs/right.jpg')))/255.

left_g = np.mean(left,axis=2)
right_g = np.mean(right,axis=2)
                   
cv = buildcv(left_g,right_g,50)
d = viterbilr(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3a.jpg'),dimg)
