## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

def histeq(X):
    pix_tot = X.size
    Y,indices,rec_ind,bincount = np.unique(X,return_index=True, return_inverse=True, return_counts=True)
    cdf = np.cumsum(bincount)
    cdf_norm = (cdf-np.amin(cdf))/(pix_tot-1)
    eq_cdf = np.round(cdf_norm*(L-1))  
    lookup = dict(zip(Y,eq_cdf))
    Z = np.zeros(X.shape)
    for ind,pixval in enumerate(X.flat):
        Z.flat[ind] = lookup.get(pixval)
    return Z

    

########################## Support code below

from os.path import normpath as fn 
import warnings
warnings.filterwarnings('ignore')

L = 256
img = imread(fn('inputs/p2_inp.png'))

out = histeq(img)

out = np.maximum(0,np.minimum(255,out))
out = np.uint8(out)
imsave(fn('outputs/prob2.png'),out)
