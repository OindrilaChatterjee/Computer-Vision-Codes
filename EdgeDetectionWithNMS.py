import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

# Different thresholds to try
T0 = 0.5
T1 = 1.0
T2 = 1.5

sobelfilt_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobelfilt_y = np.transpose(sobelfilt_x)

# Return magnitude, theta of gradients of X
def grads(X):
    Ix = conv2(X,sobelfilt_x,mode='same',boundary='symm')
    Iy = conv2(X,sobelfilt_y,mode='same',boundary='symm')
    H = np.sqrt(np.square(Ix)+np.square(Iy))
    theta = np.arctan2(Iy,Ix)
    return H,theta
 
def nms(E,H,theta):
    theta_round2 = (np.pi* np.around(4*theta/np.pi))/4
    thetagroup = np.copy(theta_round2)
    thetagroup[thetagroup<0] = np.pi + thetagroup[thetagroup<0]
    thetagroup[thetagroup == np.pi] = 0
    E0n = np.copy(E)
    Hdash = np.pad(H,((1,1),(1,1)),'constant')
    Hdashlist = np.zeros_like(Hdash)
    Hneighbor1 = np.zeros_like(Hdash)
    Hneighbor2 = np.zeros_like(Hdash)
    thetaspace = np.linspace(0,3*np.pi/4,num=4)
    movdir = [(0,1),(-1,1),(1,0),(1,1)]
    movdict = dict(zip(thetaspace,movdir))
    checkpts = np.array(np.where(E==1))
    thetalist = thetagroup[checkpts[0,:],checkpts[1,:]]
    movlist = np.transpose(np.array([movdict[x] for x in thetalist]))
    Hdashlist[checkpts[0,:]+1,checkpts[1,:]+1] = Hdash[checkpts[0,:]+1,checkpts[1,:]+1]
    Hneighbor1[checkpts[0,:]+1,checkpts[1,:]+1] = Hdash[checkpts[0,:]+1-movlist[0,:],checkpts[1,:]+1-movlist[1,:]]
    Hneighbor2[checkpts[0,:]+1,checkpts[1,:]+1] = Hdash[checkpts[0,:]+1+movlist[0,:],checkpts[1,:]+1+movlist[1,:]]
    acindices = np.array(np.where(np.logical_or(Hdashlist<Hneighbor1,Hdashlist<Hneighbor2)))
    E0n[acindices[0,:]-1,acindices[1,:]-1] = 0 
     
    return E0n



########################## Support code 

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/input_image.png')))/255.

H,theta = grads(img)

imsave(fn('outputs/gradient_image.png'),H/np.max(H[:]))

## Edge detection

E0 = np.float32(H > T0)
E1 = np.float32(H > T1)
E2 = np.float32(H > T2)

imsave(fn('outputs/output_image_0.png'),E0)
imsave(fn('outputs/output_image_1.png'),E1)
imsave(fn('outputs/output_image_2.png'),E2)

## Edge detection with NMS

E0n = nms(E0,H,theta)
E1n = nms(E1,H,theta)
E2n = nms(E2,H,theta)

imsave(fn('outputs/output_image_nms0.png'),E0n)
imsave(fn('outputs/output_image_nms1.png'),E1n)
imsave(fn('outputs/output_image_nms2.png'),E2n)
