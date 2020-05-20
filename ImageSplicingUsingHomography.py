import numpy as np


# Fits a homography between pairs of pts
#   pts: Nx4 array of (x,y,x',y') pairs of N >= 4 points
# Return homography that maps from (x,y) to (x',y')

def getH(pts):
    N = len(pts)
    pts =np.asarray(pts)
    zv = np.reshape(np.array([0,0,0]),(1,3))
    p = np.append(pts[:,0:2],np.ones((N,1)),axis=1)
    pdash = np.append(pts[:,2:4],np.ones((N,1)),axis=1) 
    A = np.zeros((3*N,9))
    for i in range(N):
        a = np.reshape(p[i,:],(1,3))
        Ai1 = np.array([[0,-pdash[i,2],pdash[i,1]],[pdash[i,2],0,-pdash[i,0]],[-pdash[i,1],pdash[i,0],0]])
        Ai2 = np.concatenate((np.concatenate((a,zv,zv),axis=1),np.concatenate((zv,a,zv),axis=1),np.concatenate((zv,zv,a),axis=1)),axis=0)
        Ai = np.matmul(Ai1,Ai2)
        A[3*i:3*i+3,:] = Ai
    U, s, V = np.linalg.svd(A)
    H = np.reshape(V.T[:,-1],(3,3))
    H = H/H[2,2]
    return H
    

# Splices the source image into a quadrilateral in the dest image,
# where dpts in a 4x2 image with each row giving the [x,y] co-ordinates
# of the corner points of the quadrilater (in order, top left, top right,
# bottom left, and bottom right).
#
# Note that both src and dest are color images.
#
# Returns a spliced color image.
def splice(src,dest,dpts):
    
    h = src.shape[0]
    w = src.shape[1]
    source = np.float32([ [0,0],[0,w-1],[h-1,0],[h-1,w-1]]) 
    pts = np.concatenate((dpts,source),axis=1)
    H = getH(pts)
    destx = np.arange(np.min(dpts[:,0]),np.max(dpts[:,0])+1)
    desty = np.arange(np.min(dpts[:,1]),np.max(dpts[:,1])+1)
    [dest_r,dest_c] = np.meshgrid(destx,desty)
    dest_rec = np.concatenate((np.reshape(dest_r,(-1,1)),np.reshape(dest_c,(-1,1))),axis=1)
    hom_destrec = np.concatenate((dest_rec,np.ones((len(dest_rec),1))),axis=1).T
    source_coord = np.matmul(H,hom_destrec)
    source_coord = source_coord[0:2,:]/np.broadcast_to(source_coord[2,:],(2,len(dest_rec)))
    tx = ((source_coord[0,:]>=0) & (source_coord[0,:]<h-1))*1
    ty = ((source_coord[1,:]>=0) & (source_coord[1,:]<w-1))*1
    t = tx*ty
    source_coordx = source_coord[0,np.where(t==1)]
    source_coordy = source_coord[1,np.where(t==1)]
    floor_x = np.floor(source_coordx).astype(int)
    ceil_x = np.ceil(source_coordx).astype(int)
    ax = source_coordx-floor_x
    floor_y = np.floor(source_coordy).astype(int)
    ceil_y = np.ceil(source_coordy).astype(int)
    ay = source_coordy-floor_y
    destx = dest_rec.T[0,np.where(t==1)].astype(int)
    desty = dest_rec.T[1,np.where(t==1)].astype(int)
    I_source = (1-ax[:,:,np.newaxis])*((1-ay[:,:,np.newaxis])*src[np.ravel(floor_x),np.ravel(floor_y),:]+ay[:,:,np.newaxis]*src[np.ravel(floor_x),np.ravel(ceil_y),:])+ax[:,:,np.newaxis]*((1-ay[:,:,np.newaxis])*src[np.ravel(ceil_x),np.ravel(floor_y),:]+ay[:,:,np.newaxis]*src[np.ravel(ceil_x),np.ravel(ceil_y),:])
    dest[np.ravel(desty),np.ravel(destx),:] = I_source

    return dest
    
    
########################## Support code 

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

def clip(im):
    return np.maximum(-1,np.minimum(1.,im))

simg = np.float32(imread(fn('inputs/source_image.png')))/255.
dimg = np.float32(imread(fn('inputs/desination_image.png')))/255.
dpts = np.float32([ [276,54],[406,79],[280,182],[408,196]]) # Hard coded

comb = splice(simg,dimg,dpts)

imsave(fn('outputs/output_image.png'),clip(comb))

