import numpy as np
from numpy.linalg import inv
from skimage.io import imread, imsave

def im2wv(img,nLev): ## implements image to wavelet transformation ##
    
    ## img : input image
    ## nLev : number of levels
    
    transmat = 0.5 * np.array([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]])
    pyr = []
    imgnew = np.copy(img)
    for lev in range(nLev):
        x,y = imgnew.shape
        L = []
        H1 = []
        H2 = []
        H3 = []
        for i in np.arange(0,x,2):
            for j in np.arange(0,y,2):
                clistrefnew = np.matmul(transmat,imgnew[i:i+2,j:j+2].flatten())
                L.append(clistrefnew[0])
                H1.append(clistrefnew[1])
                H2.append(clistrefnew[2])
                H3.append(clistrefnew[3])
        L = np.array(L).reshape(np.int32(x/2),np.int32(y/2))
        H1 = np.array(H1).reshape(np.int32(x/2),np.int32(y/2))
        H2 = np.array(H2).reshape(np.int32(x/2),np.int32(y/2))
        H3 = np.array(H3).reshape(np.int32(x/2),np.int32(y/2))
        V = [H1,H2,H3]
        pyr.append(V)
        imgnew = np.copy(L)
    pyr.append(L)
        
    return pyr


def wv2im(pyr):   ## implements wavelet to image transformation ##
    
    nLev = len(pyr)-1
    transmat = 0.5 * np.array([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]])
    invtransmat = inv(transmat)
    Lrec = pyr[-1]
    L = np.copy(Lrec)
    sz = np.array(L.shape)
    for level in range(nLev):      
        V = pyr[nLev-level-1]
        H1 = V[0]
        H2 = V[1]
        H3 = V[2]
        imgnew=[]
        xnew,ynew = L.shape
        for i in range(0,xnew,1):
            for j in range(0,ynew,1):
                multvec = np.transpose(np.array([L[i][j],H1[i][j],H2[i][j],H3[i][j]]))
                imgblock = np.matmul(invtransmat,multvec)
                imgnew.append(imgblock.reshape(2,2))
        imsize1,imsize2 = 2**(level+1)*sz
        L = np.array(imgnew).reshape(np.int32(imsize1/2),-1,2,2).swapaxes(1,2).reshape(imsize1,imsize2)   
    return L



########################## Support code 

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Clip intensities b/w 0 and 1
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


# Visualize pyramid 
def vis(pyr, lev=0):
    if len(pyr) == 1:
        return pyr[0]/(2**lev)

    sz=pyr[0][0].shape
    sz1 = [sz[0]*2,sz[1]*2]
    img = np.zeros(sz1,dtype=np.float32)

    img[0:sz[0],0:sz[1]] = vis(pyr[1:],lev+1)

    # Scale / shift gradient images for visualization
    img[sz[0]:,0:sz[1]] = pyr[0][0]*(2**(1-lev))+0.5
    img[0:sz[0],sz[1]:] = pyr[0][1]*(2**(1-lev))+0.5
    img[sz[0]:,sz[1]:] = pyr[0][2]*(2**(1-lev))+0.5

    return img



############ Main Program


img = np.float32(imread(fn('inputs/input_image.png')))/255.

# Visualize pyramids
pyr = im2wv(img,1)
imsave(fn('outputs/output_image_lev1.png'),clip(vis(pyr)))

pyr = im2wv(img,2)
imsave(fn('outputs/output_image_lev2.png'),clip(vis(pyr)))

pyr = im2wv(img,3)
imsave(fn('outputs/output_image_lev3.png'),clip(vis(pyr)))

# Inverse transform to reconstruct image
im = clip(wv2im(pyr))
imsave(fn('outputs/output_wavelet_to_image.png'),im)

# Zero out some levels and reconstruct
for i in range(len(pyr)-1):

    for j in range(3):
        pyr[i][j][...] = 0.

    im = clip(wv2im(pyr))
    imsave(fn('outputs/output_wavelet_to_image_zero_out_lev%d.png' % i),im)