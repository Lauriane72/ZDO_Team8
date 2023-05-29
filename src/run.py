#technical packages
import os
import json

#work packages
import math
import random
from matplotlib import pyplot as plt
import numpy as np
import cv2
import skimage.io
from skimage import exposure, morphology, measure
from skimage.transform import (hough_line, hough_line_peaks)

"""main code"""
def main(*images):

    for image in images:

        """FIRST PART: GET THE IMAGE READY FOR THE EVALUATION"""
        # image enhancement
        img_enhanced = exposure.equalize_adapthist(image)
        img_filtered = cv2.medianBlur(img_enhanced.astype('float32'), 3)
    
        # image segmentation
        block_size = 19
        threshold = skimage.filters.threshold_otsu(img_filtered,block_size)
        img_seg = img_filtered < threshold

        # morphological operations
        img_op = morphology.binary_opening(img_seg)
        img_morpho = morphology.area_closing(img_op)

        # object description
        label = measure.label(img_morpho, background=0)
        props = measure.regionprops_table(label, image, properties=['label','area','equivalent_diameter','mean_intensity','solidity'])
        maxval = max(props['area'])
        img_labelled = morphology.remove_small_objects(label, min_size=maxval-1)

        # skeletonisation
        img_skeleton = morphology.skeletonize(img_labelled)


    return(image)

ok=main(dataset[1])
for image in ok:
    print(ok)
