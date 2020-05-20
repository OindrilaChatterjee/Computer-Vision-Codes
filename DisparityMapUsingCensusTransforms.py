import numpy as np


#########################################
### Hamming distance computation
### Call the function hamdist with two uint32 bit arrays of the same size. It will
### return another array of the same size with the elmenet-wise hamming distance.
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


## Computes a 5x5 census transform of the grayscale image img.
## Returns a uint32 array of the same shape

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

## Given left and right image and max disparity D_max, returns a disparity map
## based on matching with  hamming distance of census codes. Uses the census function above.

## d[x,y] implies that left[x,y] matched best with right[x-d[x,y],y]. Disparity values
## should be between 0 and D_max (both inclusive).

def smatch(left,right,dmax):
    
    [H,W] = left.shape
    ham_mat = np.zeros_like(left)
    cenleft = census(left)
    cenright = census(right)
    for x in range(W-1):
        testmat = hamdist(np.reshape(cenleft[:,x],(-1,1)),cenright[:,np.maximum(0,(x-dmax)):x+1])
        ham_mat[:,x] = np.argmin(np.fliplr(testmat),axis=1)
      
    return ham_mat
    
    
########################## Support code 

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = imread(fn('inputs/left.jpg'))
right = imread(fn('inputs/right.jpg'))

d = smatch(left,right,40)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/20.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/output.png'),dimg)
