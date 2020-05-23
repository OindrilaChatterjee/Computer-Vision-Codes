import numpy as np
from scipy.signal import convolve2d as conv2

# Use the following as the x and y derivative filters
fx = np.float32([[1,0,-1]]) * np.float32([[1,1,1]]).T / 6.
fy = fx.T



# Computes optical flow using the lucas kanade method
# Uses the fx, fy, defined above as the derivative filters
# and computes derivatives on the average of the two frames.
# Also, considers (x',y') values in a WxW window.
# Returns two image shape arrays u,v corresponding to the
# horizontal and vertical flow.
def lucaskanade(f1,f2,W):
    u = np.zeros(f1.shape)
    v = np.zeros(f1.shape)
    [l,w] = np.shape(f1)
    Imid = 0.5*(f1+f2)
    It = f2-f1
    Ix = conv2(Imid,fx,mode='same',boundary='symm')
    Iy = conv2(Imid,fy,mode='same',boundary='symm')
    win = np.ones([W,W])
    eps  = 1e-12
    A11 = conv2(np.square(Ix),win,mode='same',boundary='symm')+eps
    A22 = conv2(np.square(Iy),win,mode='same',boundary='symm')+eps
    A12 = conv2(np.multiply(Ix,Iy),win,mode='same',boundary='symm')
    B1 = -conv2(np.multiply(Ix,It),win,mode='same',boundary='symm')
    B2 = -conv2(np.multiply(Iy,It),win,mode='same',boundary='symm')
    D = np.multiply(A11,A22)-np.square(A12)
    unum = np.multiply(B1,A22)-np.multiply(B2,A12)
    vnum = np.multiply(B2,A11)-np.multiply(B1,A12)
    u = np.divide(unum,D)
    v = np.divide(vnum,D)
    
    return u,v
    
########################## Support code 

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


f1 = np.float32(imread(fn('inputs/frame1.jpg')))/255.
f2 = np.float32(imread(fn('inputs/frame2.jpg')))/255.

u,v = lucaskanade(f1,f2,11)
#out = lucaskanade(f1,f2,11)


# Display quiver plot by downsampling
x = np.arange(u.shape[1])
y = np.arange(u.shape[0])
x,y = np.meshgrid(x,y[::-1])
plt.quiver(x[::8,::8],y[::8,::8],u[::8,::8],-v[::8,::8],pivot='mid')

plt.show()
