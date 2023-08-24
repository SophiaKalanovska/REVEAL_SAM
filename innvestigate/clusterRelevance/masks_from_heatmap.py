import numpy as np
from . import  one_dimensional_breaks, two_dimensional_clusters, random_clusters, masks_from_heatmap


def resize(ret, size):
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    if ret.ndim == 2:
        ret.resize((size, size, 1))
        ret = np.repeat(ret, 3, axis=-1)
    return ret     


def retrieve(a, x, size):
    # b = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    # b /= np.max(np.abs(b))
    # activations = b[0]
    # numberOfBreaks = 2
    # cutoff = 1000
    # n = 5

    # categories = one_dimensional_breaks.jenks_breaks(activations, numberOfBreaks, activations[-1].size, n, cutoff, x)

    # # Visualise the activation ranges
    # # one_dimensional_breaks.visualise_breaks(b[0])

    # # Create masks for each cluster found on the level with highest activation
    # masks_2D, masks_with_ones = two_dimensional_clusters.DbSCAN_for_activations(categories, activations[-1].size, a)

    # masks_3D = random_clusters.random_clusters(b, masks_with_ones)

    # masks = [
    #     mask['segmentation']
    #     for mask
    #     in sorted(a, key=lambda x: x['area'], reverse=True)
    # ]


    masks = [
        mask['segmentation']
        for mask
        in sorted(a, key=lambda x: x['area'], reverse=True)
    ]

    masks_resized = []
    for ret in masks:
        ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
        if ret.ndim == 2:
            ret = np.expand_dims(ret, -1)
            masks_resized.append(np.expand_dims(np.repeat(ret, 3, axis=-1), 0))

    top_6 = masks_resized[: 6]

    masks_3D = random_clusters.random_clusters(x, top_6)

    masks = masks[: 6]


    return masks, masks_3D
