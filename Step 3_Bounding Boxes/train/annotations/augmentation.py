from imgaug import augmenters as iaa
import numpy as np
import glob 
import os
import imgaug as ia
from skimage import io
from skimage.io import imsave
from skimage.transform import rescale, resize, downscale_local_mean
from random import randrange
from imgaug.augmentables.segmaps import SegmentationMapsOnImage



modules = glob.glob(os.path.join(".","*.png"))   




for imstr in modules:
        filename = os.path.join(imstr)
        image = io.imread(filename)[:,:,:3]
        image = (np.array(image)<5)*255
 
        imsave((imstr),image.astype(np.uint8) )

    