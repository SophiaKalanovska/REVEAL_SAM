import numpy as np
from . import  one_dimensional_breaks, two_dimensional_clusters, random_clusters, masks_from_heatmap

import cv2

def remove_contained_parts(bigger_mask, smaller_masks):
    # Ensure the bigger_mask is a numpy array and of dtype np.uint8
    bigger_mask = np.array(bigger_mask)
    bigger_mask = bigger_mask.astype(np.uint8)

    # Iterate through each smaller mask and remove its part from the bigger mask
    for smaller_mask in smaller_masks:
        # Ensure the smaller_mask is a numpy array and of dtype np.uint8
        smaller_mask = np.array(smaller_mask)
        smaller_mask = smaller_mask.astype(np.uint8)

        # Check if the smaller mask is contained in the bigger mask
        # if is_mask_contained(bigger_mask, smaller_mask):
            # Remove the part of the bigger mask that is contained by the smaller mask
            
        bigger_mask = cv2.bitwise_xor(bigger_mask, cv2.bitwise_and(bigger_mask, smaller_mask))

    return bigger_mask

def count_ones(mask):
    return np.count_nonzero(mask == 1)

def expand_cluster(mask, num_pixels=5):
    # Define the kernel for dilation (a square kernel with side length of num_pixels)
    kernel = np.ones((num_pixels, num_pixels), np.uint8)

    # Perform dilation on the mask
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    return dilated_mask


def retrieve_pixels(a, x, size):
    b = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    b /= np.max(np.abs(b))
    activations = b[0]
    numberOfBreaks = 2
    cutoff = 1000
    n = 5

    categories = one_dimensional_breaks.jenks_breaks(activations, numberOfBreaks, activations[-1].size, n, cutoff, x)

    # Visualise the activation ranges
    # one_dimensional_breaks.visualise_breaks(b[0])

    # Create masks for each cluster found on the level with highest activation
    masks, masks_with_ones = two_dimensional_clusters.DbSCAN_for_activations(categories, activations[-1].size, a)

    masks_3D = random_clusters.random_clusters(b, masks_with_ones)

    return masks, masks_3D

def retrieve(a, x, size):
    masks = [
        mask['segmentation']
        for mask
        in sorted(a, key=lambda x: x['area'], reverse=True)
    ][:10]

    

    num_masks = len(masks)
    # containment_matrix = np.zeros((num_masks, num_masks), dtype=bool)
    masks_two_channel = []
    # for i in range(num_masks):
    #     num_pixels = x.shape[1]/ 50
    #     masks[i] = expand_cluster(masks[i], np.ceil(num_pixels).astype(np.int32))

    # for i in range(num_masks):
    #     masks[i] = remove_contained_parts(masks[i], masks[i+1:])
    #     num_pixels = x.shape[1]/ 100
    #     masks[i] = expand_cluster(masks[i], np.ceil(num_pixels).astype(np.int32))

    # masks = sorted(masks, key=count_ones, reverse=True)[: 6]
    num_masks = len(masks)
    for i in range(num_masks):
        masks[i] = remove_contained_parts(masks[i], masks[i+1:])     
        masks_two_channel.append(masks[i].astype(np.bool_))

    
    masks_resized = []
    for ret in masks:
        ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
        if ret.ndim == 2:
            ret = np.expand_dims(ret, -1)
            masks_resized.append(np.expand_dims(np.repeat(ret, 3, axis=-1), 0))


    
    masks_resized, masks_two_channel = zip(*sorted(zip(masks_resized, masks_two_channel), key = lambda x: count_ones(x[0]), reverse=True))
    masks_resized = list(masks_resized)
    masks_two_channel = list(masks_two_channel)
    top_6 = masks_resized[: 6]

    masks_3D = random_clusters.random_clusters(x, top_6)

    masks = masks_two_channel[: 6]


    return masks, masks_3D
