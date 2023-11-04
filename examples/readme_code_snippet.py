# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:

from __future__ import \
    absolute_import, print_function, division, unicode_literals

import numpy as np
np.random.seed(42)
# import innvestigate as inn
from importlib.machinery import SourceFileLoader
import copy
import random
# from clusterRelevance import illustrate_clusters
# from clusterRelevance import masks_from_heatmap
# from sklearn.preprocessing import StandardScaler
# from tensorflow.python.framework.opscd .. import enable_eager_execution_internal
import os
import sys
import torch
from segment_anything import sam_model_registry
import cv2
from segment_anything import SamAutomaticMaskGenerator
import supervision as sv


file_dir = os.path.dirname(__file__)

sys.path.append(file_dir)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import innvestigate
import innvestigate.utils
# from innvestigate.utils import *
import tensorflow.keras.applications.inception_v3 as inception
import keras.applications.vgg16 as vgg16
import keras.applications.resnet as res
import keras.applications.vgg19 as vgg19

import keras.applications.inception_v3 as inception

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import argparse
import pickle


###############################################################################
###############################################################################

base_dir = os.path.dirname(__file__)
utils = SourceFileLoader("utils", os.path.join(base_dir, "utils.py")).load_module()

###############################################################################
###############################################################################

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    # Load an image.
    # Need to download examples images first.
    # See script in images directory.

    # folder_path = base_dir + "/ILSVRC"  # Modify this as needed

    # List all files in the folder
    # filenames = os.listdir(folder_path)
    # if len(sys.argv) < 2:
    #     print("Usage: python my_program.py <image_path>")
    #     sys.exit(1)

    # image_path = sys.argv[1]
    image_path = "SAM1.JPEG"

    # Process each filename using the method

    # image_name = filename
    # image_name = "tiger.jpg"
    # image_name = "guini.jpg"
    # image_name = "piano_monkey.jpeg"
    # image_name = "bike_balloons.jpeg"
    print(image_path)


    # image_size = 224
    image_size = 299
    image = utils.load_image(
        os.path.join(base_dir, "ILSVRC", image_path), image_size)
    image_new = image[:, :, :3]

    CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(sam)
    IMAGE_PATH = base_dir + "/images/"+ image_path

    image_bgr = np.asarray(image_new, dtype=np.uint8)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = mask_generator.generate(image_rgb)



    mask_annotator = sv.MaskAnnotator()
    detections = sv.Detections.from_sam(result)

    # model_call = vgg16
    # model_call = vgg19
    model_call = inception
    # model_call = res

    # model, preprocess = vgg16.VGG16(), vgg16.preprocess_input
    # model, preprocess = res.ResNet50(), res.preprocess_input
    # model, preprocess = vgg19.VGG19(), vgg19.preprocess_input
    model, preprocess = inception.InceptionV3(), inception.preprocess_input

    # Strip softmax layerexamples

    model = innvestigate.model_wo_softmax(model)
    x = preprocess(image[None])

    masks, masks_from_heatmap3D = innvestigate.masks_from_heatmap.retrieve(result, x, image_size)


    # Get model

    # Create analyzer

    # Add batch axis and preprocess


    analyzer = innvestigate.create_analyzer("lrp.alpha_1_beta_0", model)
    pr = model.predict_on_batch(x)
    the_label_index = np.argmax(pr, axis=1)
    predictions = model_call.decode_predictions(pr)

    # # distribute the relevance to the input layer
    start_time = time.time()
    a = analyzer.analyze(x)
    # print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    # # Aggregate along color channels and normalize to [-1, 1]
    # b = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    # b /= np.max(np.abs(b))

    # # # Plot
    # plt.imshow(b[0], cmap="seismic", clim=(-1, 1))
    # plt.savefig("vgg16_heat_map/" + image_path + "_heatmap.png")
    # # plt.show()
    # plt.clf()
    # # 


    # Plot in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # x = np.linspace(0, 224, 224)
    # y = np.linspace(0, 224, 224)
    # X, Y = np.meshgrid(x, y)
    # Z = b[0]

    # ax.plot_surface(X, Y, Z, cmap='viridis')

    # plt.show()

    masks_pixels, masks_from_heatmap3D_pixels = innvestigate.masks_from_heatmap.retrieve_pixels(a, x, image_size, image_path)

    sorted_mask, sorted_masks_3D = innvestigate.masks_from_heatmap.rank_and_sort_masks(masks, masks_from_heatmap3D, masks_from_heatmap3D_pixels)
    
    sorted_mask, sorted_masks_3D = sorted_mask[:10], sorted_masks_3D[:10]

    analyzer = innvestigate.create_analyzer("reveal.alpha_2_beta_1", model, **{"masks": masks_from_heatmap3D_pixels})


    # # # Apply analyzer w.r.t. maximum activated output-neuron
    # start_time = time.time()
    relevance = analyzer.analyze(x)
    # print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    illustrate = innvestigate.illustrate_clusters.Illustrate()
    illustrate.mask_to_input_relevance_of_mask([random.randint(0, 100) for _ in range(len(sorted_mask)+2)], sorted_masks_3D, scene_colour = copy.copy(image_rgb), detections= detections, masks = sorted_mask)
    # illustrate.mask_to_input_relevance_of_pixels([random.randint(0, 100) for _ in range(len(masks_pixels)+2)], masks_from_heatmap3D_pixels, label = predictions[0][0][1], image_name= image_path)