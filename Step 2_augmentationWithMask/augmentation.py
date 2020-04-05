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


nbrOfAugCopies = 20

bgImages = glob.glob(os.path.join(".","Backgrounds","*.jpg"))   

modules = glob.glob(os.path.join(".","Input","*.png"))   

i=0

image = io.imread(modules[0])[:,:,:3]
sizeImages = (np.shape(image)[0],np.shape(image)[1])


for imstr in modules:
    for j in range(nbrOfAugCopies):
        filename = os.path.join(imstr)
        image = io.imread(filename)[:,:,:3]
#'''
        randBG = randrange(0, len(bgImages), 1) 
        bg  = io.imread(bgImages[randBG])
        seq = iaa.Sequential([iaa.Crop(percent=(0, 0.3)),
            iaa.Fliplr(0.5), ], random_order=True) 

        bg = seq(image=bg.astype(np.uint8))
        bg = resize(bg, sizeImages,anti_aliasing=True)*255


        seq = iaa.Sequential([iaa.Sometimes(0.3,iaa.Cutout(nb_iterations=(1, 7),size=1/8,fill_mode="constant", cval=0)),
            iaa.Sometimes(0.3,iaa.Crop(percent=(0, 0.2))),
            iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0, 0.3))),
            iaa.Sometimes(0.2,iaa.Dropout(p=(0, 0.05))),
            iaa.LinearContrast((0.7, 2)),
            iaa.Sometimes(0.7,iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,0.1*255), per_channel=0.2))], random_order=True) 

        seq2 = iaa.Sequential([iaa.Affine(rotate=(-125, 125),shear=(-8, 8))], random_order=True) 
        image = seq2(image=image.astype(np.uint8))

        mask = (image<25)*1
        segmap = SegmentationMapsOnImage(mask.astype(np.uint8), shape=image.shape)


        image__=(bg*mask) + image
        images_aug,mask  = seq(image=image__.astype(np.uint8),segmentation_maps=segmap)
        
        mask = np.array(mask.draw())[1,:,:,:]
        mask = (mask<2)*255
        print(np.max(mask))
        
#        imsave(("Augmented/"+str(i+1000)+".jpg"),images_aug )
#        imsave(("Masks/"+str(i+1000)+"_beet_"+str(i)+".png"),mask )
        imsave(("train/shapes_train2018/"+str(i+1000)+".jpg"),images_aug.astype(np.uint8) )
        imsave(("train/annotations/"+str(i+1000)+"_beet_"+str(i)+".png"),mask.astype(np.uint8) )

        

        print("doing the : " + str(i+1)+"/"+str(len(modules)*nbrOfAugCopies))
        i+=1
#	imsave(imstr,images_aug)

#'''