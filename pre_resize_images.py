# one-off script to resize images to k*512, in order to reduce their size before uploading to cloud service

# ARGUMENTS
path_to_db = "../../../data/" # path to old database
path_to_new_db = "/tmp/data/" # path to new database
thres_pixels = 700 # thresshold larger than which, the images will be downscale to

import os
import shutil
import sys

# path to the home-directory, working directory, and repository code (to import)
home_path = os.path.expanduser('~') # get home path
path_to_wd = "Documents/ScriptsPrograms/ml_art/bin/gan_zoo_pytorch/pytorch_GAN_zoo/"
path_to_repo_code = "Documents/ScriptsPrograms/ml_art/bin/gan_zoo_pytorch/pytorch_GAN_zoo/"

# change home directory; add repositry code directory to sys.path
os.chdir(os.path.join(home_path, path_to_wd))
sys.path.append(os.path.join(home_path, path_to_repo_code))

import importlib
#import argparse
import json
import re

from models.utils.utils import getVal, getLastCheckPoint, loadmodule
from models.utils.config import getConfigOverrideFromParser, updateParserWithConfig
import numpy as np

import albumentations as A
import cv2
import magic
from models.utils.image_transform import pil_loader, ToTensorV2

def get_img_dim(path_to_img):
    """uses regex and the output from magic module to extract the text description of image size; return number dimensions"""
    img_info_str = magic.from_file(path_to_img)
    # find', 1234x1234, '
    #m =re.search('\,\s\d+x\d+\,\s',img_info_str)
    m = re.search('\,\s\d+\s*x\s*\d+\,\s',img_info_str)
    if not bool(m):
        raise ValueError("""no resolution found in %s""" % img_info_str)
    # split by x
    res_xy_str  =re.sub('[\s\,]','',m.group()).split('x')
    # make into integers
    res = [int(x) for x in re.sub('[\s\,]','',m.group()).split('x')]
    return res

def check_image_requires_resizing(image_dimensions, threshold):
    """check whether any dimenion is lower than the mandated threshold"""
    return all(list(map(lambda l: l>threshold, image_dimensions)))

def new_dimension( image_dimension, threshold):
    """what is the new dimension of the resized image"""
    min_dim = min(image_dimension)
    idx_min_dim = image_dimension.index(min_dim)
    # get the scale necessary to rescale 
    scale_ = threshold/image_dimension[idx_min_dim]
    # downscaled dimension
    dim_new = [int(d*scale_) for d in image_dimension]
    return dim_new

def downscale(path_to_image, new_dimension, path_to_new_image = None, plot=None):
    """ use albumentations to downscale an image"""
    if plot is None:
        plot = False
    
    img = cv2.imread(path_to_image)
    # ensure proper color
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_image = cv2.resize(img, tuple(new_dimension), interpolation=cv2.INTER_AREA)
    # whether to save an image
    if not (path_to_new_image is None):
        cv2.imwrite(path_to_new_image,new_image)
    
    # whether to plot an image
    if plot:
        plt.imshow(img); plt.show()
    
    return new_image

# make the new path
if not os.path.isdir(path_to_new_db):
    os.mkdir(path_to_new_db)

# make the subdirectories (if they don't exist)
subdirs = [d for d in os.listdir(path_to_db)]
for subdir_ in subdirs:
    if not os.path.isdir(os.path.join(path_to_new_db,subdir_)):
        os.mkdir(os.path.join(path_to_new_db,subdir_))

# get all the files, and their dimensions
paths_to_files = []
for subdir_ in subdirs:
    for f in os.listdir(os.path.join(path_to_db,subdir_)): #
        # paths, old and new
        old_path_ = os.path.join(path_to_db,subdir_,f)
        new_path_ = os.path.join(path_to_new_db,subdir_,f)
        # get dimensions
        dimensions_ = get_img_dim(old_path_)
        paths_to_files.append((old_path_, new_path_, dimensions_))

# loop through files: copy the ones that are 
for path_old, path_new, old_dimension in paths_to_files:
    # check whether an image needs rescaling
    if check_image_requires_resizing(image_dimensions=old_dimension, threshold=thres_pixels):
        dim_new = new_dimension( old_dimension, thres_pixels) # new dimension
        # new image
        new_image = downscale(path_old, dim_new, path_to_new_image = path_new, plot=False)
    else:
        shutil.copyfile(path_old, path_new) 

################
# manually flip some of the abstract paintings vertically/horizontally
images_to_flip_horizontally = ["../../../data/impression_pureabstract/0a10e4a15632b0939ce80ab517839a85.jpg","../../../data/impression_pureabstract/2ab9b409793acca1ea7ba805c5cd9f28.jpg","../../../data/impression_pureabstract/2A71A49D-7F06-4D35-AF5D-5DA576743FF4.JPG","../../../data/impression_pureabstract/2cde25c633bef8c165bd5a4c2cbbfd61.jpg","../../../data/impression_pureabstract/5d6c310f1a2c8b801bef020b84f42a42.jpg","../../../data/impression_pureabstract/6a78d139c5c8ea726e2a5ab71d4730b5.jpg","../../../data/impression_pureabstract/8dac4ef6862a364d2bcfbdfa4a336433.jpg","../../../data/impression_pureabstract/8x10czkqmse31.jpg","../../../data/impression_pureabstract/395d6f26148443.5635059799b28.jpg","../../../data/impression_pureabstract/589a826e03b972eba916577be422c7ee.jpg","../../../data/impression_pureabstract/10186%20-%20Aloft%2030%2008%2018.JPG","../../../data/impression_pureabstract/10204%20-%2020%2011%2018.JPG","../../../data/impression_pureabstract/10205%20-%20Storm%20Signed%2021%2011%2018.JPG","../../../data/impression_pureabstract/10207%20-%20Downpour%20-%2022%2011%2018%20signed.JPG","../../../data/impression_pureabstract/10210%20-%20Converge%2027%2011%2018.jpg","../../../data/impression_pureabstract/105148_1.jpg","../../../data/impression_pureabstract/60045646_616876865495823_1421327681814790144_o.jpg","../../../data/impression_pureabstract/109898162_3134160280013618_5519726051067852188_o.jpg","../../../data/impression_pureabstract/110179113_3134160306680282_3556583195638535398_o.jpg","../../../data/impression_pureabstract/ae4cc7ca51af703094311b9055e83db1.jpg","../../../data/impression_pureabstract/a-place-to-be-i-main.jpg","../../../data/impression_pureabstract/art2.png","../../../data/impression_pureabstract/art3.png","../../../data/impression_pureabstract/art6.png","../../../data/impression_pureabstract/art7.png","../../../data/impression_pureabstract/art12.png","../../../data/impression_pureabstract/art15.png","../../../data/impression_pureabstract/art21.png","../../../data/impression_pureabstract/art24.png","../../../data/impression_pureabstract/art29.png","../../../data/impression_pureabstract/art37.png","../../../data/impression_pureabstract/art45.png","../../../data/impression_pureabstract/art48.png","../../../data/impression_pureabstract/blue-dusk.jpg","../../../data/impression_pureabstract/c2c78a3c83fb610431215fae96e52e6f.jpg","../../../data/impression_pureabstract/e6e276e56ee83a3c96f12038fbf33cc7.jpg","../../../data/impression_pureabstract/H0649-L13590363.jpg","../../../data/impression_pureabstract/iheartheclockitsticking-798x800.jpg","../../../data/impression_pureabstract/salary-scale-ulla-maria-johanson-2016-64b4b10b.jpg","../../../data/impression_pureabstract/Tao Triptych.jpg","../../../data/impression_pureabstract/tumblr_o05b7yy7cn1r9594zo1_540.jpg","../../../data/impression_pureabstract/veitart_baoha_sm_red.JPG","../../../data/impression_pureabstract/vietart_baoha_crap2.JPG","../../../data/impression_pureabstract/vietart_baoha_sm_blue.JPG"]

images_to_flip_vertically = ["../../../data/impression_pureabstract/0a10e4a15632b0939ce80ab517839a85.jpg","../../../data/impression_pureabstract/2A71A49D-7F06-4D35-AF5D-5DA576743FF4.JPG","../../../data/impression_pureabstract/2ab9b409793acca1ea7ba805c5cd9f28.jpg","../../../data/impression_pureabstract/2cde25c633bef8c165bd5a4c2cbbfd61.jpg","../../../data/impression_pureabstract/5d6c310f1a2c8b801bef020b84f42a42.jpg","../../../data/impression_pureabstract/240.jpg","../../../data/impression_pureabstract/386_7__DxO.jpg",'../../../data/impression_pureabstract/2237bfc6dce338f486b176a123698968.jpg',"../../../data/impression_pureabstract/10210%20-%20Converge%2027%2011%2018.jpg","../../../data/impression_pureabstract/398109w550.jpg","../../../data/impression_pureabstract/60045646_616876865495823_1421327681814790144_o.jpg","../../../data/impression_pureabstract/Abstract-Oil-painting-Melody-for-Guitar-and-Sax-Silvia-Vassileva-Painting-Modern-Canvas-art-Room-decor.jpg","../../../data/impression_pureabstract/110318719_3134160330013613_2888548261907595138_o.jpg","../../../data/impression_pureabstract/acumen_by_narcisse_shrapnel_d2sibni-fullview.jpg","../../../data/impression_pureabstract/animax0.jpg","../../../data/impression_pureabstract/a-place-to-be-i-main.jpg","../../../data/impression_pureabstract/art.png","../../../data/impression_pureabstract/art3.png","../../../data/impression_pureabstract/art5.png","../../../data/impression_pureabstract/art7.png","../../../data/impression_pureabstract/art8.png","../../../data/impression_pureabstract/art11.png","../../../data/impression_pureabstract/art25.png","../../../data/impression_pureabstract/art26.png","../../../data/impression_pureabstract/art37.png","../../../data/impression_pureabstract/art38.png","../../../data/impression_pureabstract/art40.png","../../../data/impression_pureabstract/art43.png","../../../data/impression_pureabstract/art46.png","../../../data/impression_pureabstract/art49.png","../../../data/impression_pureabstract/art51.png","../../../data/impression_pureabstract/art53.png","../../../data/impression_pureabstract/art54.png","../../../data/impression_pureabstract/aYXHay5.jpg","../../../data/impression_pureabstract/d190cca1c8d080e6fcb389a416d3e297.jpg","../../../data/impression_pureabstract/Echoes of Summer 80cm x 80cm Oil on Canvas main.jpg","../../../data/impression_pureabstract/ef78191a3e731c038ead5b55b11e1cba.jpg","../../../data/impression_pureabstract/iheartheclockitsticking-798x800.jpg","../../../data/impression_pureabstract/Midnight Sun16 x 16 Oil on Constructed Aluminium Panel.jpg","../../../data/impression_pureabstract/morning-fjord-silvia-vassileva.jpg","../../../data/impression_pureabstract/Reflections-in-Red-3.jpg","../../../data/impression_pureabstract/street-sketch.jpg","../../../data/impression_pureabstract/veitart_baoha_sm_red.JPG","../../../data/impression_pureabstract/vietart_baoha_crap2.JPG","../../../data/impression_pureabstract/vietart_baoha_sm_blue.JPG","../../../data/impression_pureabstract/White-Reflections-3.jpg","../../../data/impression_pureabstract/williamWray-Conflagrant24x18.jpg"]

def downscale_and_rotate(path_to_image, threshold, flip=None, rotate=None, path_to_new_image = None, plot=None):
    """ use albumentations to downscale an image"""
    if plot is None:
        plot = False
    if flip is None:
        flip = False
    if rotate is None:
        rotate = False
    # get the old dim
    dim_old = get_img_dim(path_to_image)
    # get the image
    img = cv2.imread(path_to_image)
    
    # do flip?
    if flip:
        instruction = cv2.ROTATE_180
        img_rt = cv2.rotate(img, instruction)
        suffix_ = '_flp'
    
    if rotate:
        # random rotation 270 or 90
        if (hash(path_to_image) % 2) ==0:
            instruction = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            instruction = cv2.ROTATE_90_CLOCKWISE
        img_rt = cv2.rotate(img, instruction)
        # rotate the dimension
        dim_old = dim_old[::-1]
        # make suffix
        suffix_ = '_rt'
    
    # check if the image needs to be resized, as well
    if check_image_requires_resizing(image_dimensions=dim_old, threshold=threshold):
        dim_new = new_dimension( dim_old, threshold)
        img_rt = cv2.resize(img_rt, tuple(dim_new), interpolation=cv2.INTER_AREA)
    
    # whether to save an image
    if not (path_to_new_image is None):
        path_split1,ext_ = os.path.splitext(path_to_new_image)
        # add suffix
        path_to_new_image_rt = path_split1 + suffix_ + ext_
        cv2.imwrite(path_to_new_image_rt, img_rt)
    
    # whether to plot an image
    if plot:
        plt.imshow(img_rt); plt.show()
    
    return img_rt

# flip and rotate images
for f in images_to_flip_vertically:
    f_new = [path[1] for path in paths_to_files if path[0] == f][0]
    _ = downscale_and_rotate(path_to_image=f, threshold = thres_pixels, flip=True, rotate=False, path_to_new_image = f_new, plot=False)

for f in images_to_flip_horizontally:
    f_new = [path[1] for path in paths_to_files if path[0] == f][0]
    _ = downscale_and_rotate(path_to_image=f, threshold = thres_pixels, flip=False, rotate=True, path_to_new_image = f_new, plot=False)
    

# play with opening and saving images in ONE color channel format
# i = 0
# i+=1
# path_old, path_new, old_dimension = paths_to_files[i]   #
# path_new_1 = re.sub("/tmp/data/impression_pureabstract/", "/tmp/foo/cv_", path_new)
# path_new_2 = re.sub("../../../data/impression_pureabstract/", "/tmp/foo/orig_", path_old)
# path_new_3 = re.sub("../../../data/impression_pureabstract/", "/tmp/foo/cov_", path_old)

# # copy over the original
# shutil.copyfile(path_old, path_new_2)

# # read in for saving
# img = cv2.imread(path_old)
# cv2.imwrite(path_new_1,img)

# # convert color and save
# img2 = copy.copy(img)
# #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# img2 = cv2.resize(img2, tuple([int(i*0.8) for i in rev(img2.shape[0:2])]), interpolation=cv2.INTER_AREA)
# cv2.imwrite(path_new_3,img2)

# list of (abstract) paintings that can be flipped either horizontally or vertically
