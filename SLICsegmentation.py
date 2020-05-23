import numpy as np
from scipy.signal import convolve2d as conv2

def get_cluster_centers(im,num_clusters):
    ## Implements a method that returns an initial grid of cluster centers. First
    ## create a grid of evenly spaced centers (hint: np.meshgrid), and then make sure no centers are initialized on a sharp boundary.
    cluster_centers = np.zeros((num_clusters,2),dtype='int')

    [h,w,c] = im.shape
    k = num_clusters
    s = np.round(np.sqrt((h*w)/k))
    s_row = np.round(w/s)    #clusters per row
    s_col = np.round(h/s)     #clusters per col
    border_row = np.int(0.5*(w - (s_row-1)*s))
    border_col = np.int(0.5*(h - (s_col-1)*s))
    x = np.linspace(border_row,w-border_row+1,s_row,endpoint=True,dtype=int)
    y = np.linspace(border_col,h-border_col+1,s_col,endpoint=True,dtype=int)
    cluster = np.meshgrid(y,x,indexing='ij')
    cluster_centers[:,0] = np.asarray(cluster[0]).flatten()
    cluster_centers[:,1] = np.asarray(cluster[1]).flatten()
    
    ## check whether initial cluster centers fall on any edge
    im_grad = get_gradients(im)
    for num in range(k):
        yck = cluster_centers[num,0]
        xck = cluster_centers[num,1]
        neighbor = im_grad[yck-1:yck+2,xck-1:xck+2]
        update_center = np.asarray(np.unravel_index(neighbor.argmin(), neighbor.shape))
        cluster_centers[num,:] += np.asarray(update_center)-1

    return cluster_centers



def slic(im,num_clusters,cluster_centers):
    ## Implements the SLIC function such that all pixels assigned to a label
    ## should be close to each other in squared distance of augmented vectors.
    ## We can weight the color and spatial components of the augmented vectors
    ## differently by using the spatial_weight variable.
    cbase = np.copy(cluster_centers)
    [h,w,c] = im.shape
    grid = np.indices((h,w),dtype=int)
    yind = grid[0]
    xind = grid[1]
    s = np.int(np.round(np.sqrt((h*w)/num_clusters)))
    alpha = 1.5
    augim = np.dstack((im,alpha*yind,alpha*xind))  #augmented image
    mu = augim[cluster_centers[:,0],cluster_centers[:,1],:]
    Label = np.random.randint(num_clusters,size=(h,w))
    Labelprev = np.copy(Label)
    min_dist = np.inf*np.ones((h,w))
    clustshift = 1

    while (clustshift>0):
        
        for c in range(num_clusters):        
            [clustery,clusterx] = cluster_centers[c,:]
            ymin = max(0,clustery-s)
            ymax = min(clustery+s,h-1)
            xmin = max(0,clusterx-s)
            xmax = min(clusterx+s,w-1)
            sub_augim = augim[ymin:ymax+1,xmin:xmax+1,:] 
            sub_lab = Label[ymin:ymax+1,xmin:xmax+1]
            distsub = min_dist[ymin:ymax+1,xmin:xmax+1]
            muk = np.broadcast_to(mu[c,:],sub_augim.shape)
            newdist = np.sum((sub_augim-muk)**2,axis=2)
            sub_lab[distsub>newdist] = c
            Label[ymin:ymax+1,xmin:xmax+1] = sub_lab
            min_dist[ymin:ymax+1,xmin:xmax+1] = np.minimum(distsub,newdist)
            loc = np.where(Label==c)
            superpixc = augim[loc]
            mu[c,:] = np.mean(superpixc,axis=0)
            cluster_centers[c,:] = [np.int(mu[c,3]),np.int(mu[c,4])]
            
        clustshift = np.sum(np.not_equal(Label,Labelprev)*1)
        print(clustshift)
        Labelprev = np.copy(Label)
        
                  
    return Label

########################## Support code 

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# Use get_gradients to get the gradient of your image when initializing cluster centers.
def get_gradients(im):
    if len(im.shape) > 2:
        im = np.mean(im,axis=2)
    df = np.float32([[1,0,-1]])
    sf = np.float32([[1,2,1]])
    gx = conv2(im,sf.T,'same','symm')
    gx = conv2(gx,df,'same','symm')
    gy = conv2(im,sf,'same','symm')
    gy = conv2(gy,df.T,'same','symm')
    return np.sqrt(gx*gx+gy*gy)

# normalize_im normalizes our output to be between 0 and 1
def normalize_im(im):
    im += np.abs(np.min(im))
    im /= np.max(im)
    return im

# create an output image of our cluster centers
def create_centers_im(im,centers):
    for center in centers:
        im[center[0]-2:center[0]+2,center[1]-2:center[1]+2] = [255.,0.,255.]
    return im

im = np.float32(imread(fn('inputs/input_image.jpg')))
num_clusters = [25,49,64,81,100]
for num_clusters in num_clusters:
    cluster_centers = get_cluster_centers(im,num_clusters)
    imsave(fn('outputs/output_img1_' + str(num_clusters)+'_centers.jpg'),normalize_im(create_centers_im(im.copy(),cluster_centers)))
    out_im = slic(im,num_clusters,cluster_centers)

    Lr = np.random.permutation(num_clusters)
    out_im = Lr[np.int32(out_im)]
    dimg = cm.jet(np.minimum(1,np.float32(out_im.flatten())/float(num_clusters)))[:,0:3]
    dimg = dimg.reshape([out_im.shape[0],out_im.shape[1],3])
    imsave(fn('outputs/output_img2_'+str(num_clusters)+'.jpg'),normalize_im(dimg))
