# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:

from __future__ import \
    absolute_import, print_function, division, unicode_literals

import numpy as np
np.random.seed(42)
import tensorflow.keras.models as kmodels
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
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


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

import tensorflow as tf
# import tensorflow_datasets as tfds


@tf.function
def compute_scaled_jacobian(z_k, s_k, x):
    """
    Computes the scaled Jacobian matrix of z_k with respect to x, scaled by s_k.
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        z_k_val = z_k(x)
    jacobian = tape.jacobian(z_k_val, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    scaled_jacobian = tf.einsum('ijk,ik->ijk', jacobian, s_k)
    return scaled_jacobian

def compute_element_wise_product(a_j, scaled_jacobian):
    """
    Computes the element-wise product of activation matrix and scaled Jacobian.
    """
    return a_j[:, tf.newaxis] * scaled_jacobian

def compute_relevance_proportion(R_jk, R_j):
    """
    Computes the proportion of relevance each neuron in a layer contributed.
    """
    return R_jk / tf.reduce_sum(R_jk, axis=-1, keepdims=True)

def propagate_relevance_to_next_layer(P_jk, R_j_prime):
    """
    Computes the relevance to be distributed from one layer to the next.
    """
    return tf.einsum('ijk,ik->jk', P_jk, R_j_prime)

# @tf.function
# def forward_relevance_propagation(model, example_image, analysis):
#     """
#     Propagates relevance through the layers of the model.
#     """
#     new_relevance_j = tf.cast(tf.convert_to_tensor(analysis[0]), tf.float32)

#     return new_relevance_j  # Reverse the list to match the layer order


def compute_jacobian(activations, model, layer_index, s_k):
    """
    Compute the scaled Jacobian matrix for a specific layer.
    """
    with tf.GradientTape() as tape:
        tape.watch(activations)
        # Forward pass to compute the intermediate activations
        intermediate_output = model.layers[layer_index](activations)
    
    # Compute the Jacobian
    jacobian = tape.jacobian(intermediate_output, activations)

    elements_list = s_k.shape.as_list()

    for i in range(len(jacobian.shape) - len(elements_list)):
        elements_list.append(1)

    jacobian = tf.multiply(jacobian, tf.cast(tf.reshape(s_k, elements_list), tf.float32))
    return jacobian




def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label
###############################################################################
###############################################################################

base_dir = os.path.dirname(__file__)
utils = SourceFileLoader("utils", os.path.join(base_dir, "utils.py")).load_module()

###############################################################################
###############################################################################

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()

    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Reshape for the CNN (adding channel dimension)
    train_images = train_images[..., np.newaxis]
    test_images = test_images[..., np.newaxis]

    # It can be used to reconstruct the model identically.
    model = tf.keras.models.load_model("my_model.keras")

    # Let's check:
    example_image = test_images[0:1]  # Selecting the first image
    # plt.imshow(example_image[0, :, :, 0], cmap='gray')
    # np.testing.assert_allclose(
    #     model.predict(example_image), reconstructed_model.predict(example_image)
    # )

    model.predict(example_image)
    # plt.show()

    model_wo_softmax = innvestigate.model_wo_softmax(model)

    # Create analyzer
    analyzer = innvestigate.create_analyzer("lrp.alpha_1_beta_0", model_wo_softmax)

    # Apply analyzer w.r.t. maximum activated output-neuron
    analysis = analyzer.analyze(example_image)

    # Visualize the result

    num_layers = len(model.layers)
    R_prime = analysis[1:]  # Initial relevance

    # percentile_95 = np.percentile(analysis[1:], 95)

    # # Select elements that are at or above the 95th percentile
    # top_95_percent = arr[arr >= percentile_95]

    # P_jk = analysis[0]
    

    # graph2 = tf.Graph()
    # with graph2.as_default():
    new_relevance_j = []
    J_kj_list = []
    new_relevance_j.append(tf.cast(tf.convert_to_tensor(analysis[0]), tf.float32))
    a_j = tf.cast(tf.convert_to_tensor(example_image), tf.float32)

    for j in range(num_layers):  # Iterate backwards through layers
        R_j = analysis[-(j+1)]
        R_k = analysis[-(j+2)]
        # Compute scaled Jacobian matrix
        

        # intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.layers[j].output)
        # a_k = tf.convert_to_tensor(intermediate_model.predict(example_image))

        intermediate_layer = model.get_layer(index=j)
        a_k = intermediate_layer(a_j)
        

        s_k = tf.math.divide_no_nan(R_k, a_k)
        J_kj_list.append(s_k)

        J_kj = compute_jacobian(a_j, model, j, s_k)
       
        
        perm_from_kj_to_jk =[]
        perm_from_jk_to_kj =[]
        size_sk = len(s_k.shape)     
        size_Rj = len(R_j.shape)     
        
        for i in range(len(R_j.shape)):
            index = size_sk+ i
            perm_from_kj_to_jk.append(index)  

        for i in range(len(s_k.shape)):
            index = size_Rj+ i
            perm_from_kj_to_jk.append(i) 
            perm_from_jk_to_kj.append(index)  

        for i in range(len(R_j.shape)):   
            perm_from_jk_to_kj.append(i)   


        # A_jk =  tf.math.multiply(a_j, tf.ones_like(J_kj))
        A_kj = tf.math.multiply(a_j, tf.ones_like(J_kj))
        A_jk =  tf.transpose(A_kj, perm_from_kj_to_jk)

        # Step 3: Element-wise product and division
        R_jk = tf.math.multiply(A_jk, tf.transpose(J_kj, perm_from_kj_to_jk))

        elements_list = list(R_j.shape)

        for i in range(len(R_jk.shape) - len(elements_list)):
            elements_list.append(1)

        R_j_reshaped = tf.reshape(R_j, elements_list)
        P_jk = tf.math.divide_no_nan(R_jk, R_j_reshaped)
        
        # Step 4: Compute new relevances
        new_relevance_k = tf.tensordot(tf.transpose(P_jk, perm_from_jk_to_kj), new_relevance_j[-1], int(len(R_j.shape)))
        a_j = a_k
        new_relevance_j.append(new_relevance_k)

    pendulum_model = kmodels.Model(
            inputs=model.inputs,
            outputs=new_relevance_j,
            name=f"pendulum_analyzer_model",
        )
    ret = pendulum_model.predict_on_batch(example_image)
    # with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph()) as sess:
    #     # initilase global variables
    #     sess.run(tf.compat.v1.global_variables_initializer())
    #     input_tensor = model.input
    #     # output_tensor = model.output
    #     output_tensor = new_relevance_j
    #     result = sess.run(output_tensor, feed_dict ={input_tensor:  example_image})
    #     print("New Relevance J Values: ", new_relevance_j.eval())

    print(ret)
    # with tf.compat.v1.Session() as sess:

    #      # This will run the operations that compute the tensor and fetch its value
    #     numpy_array = sess.run(new_relevance_j)
    #     # Now you can access elements like you would in a numpy array
    #     print(numpy_array)
    # b = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
    # b /= np.max(np.abs(b))

    # plt.imshow(b, cmap="seismic", clim=(-1, 1))
    # plt.savefig("vgg16_heat_map/"  + "_heatmap.png")
    # plt.show()

    # plt.imshow(analysis.squeeze(), cmap='seismic', clim=(-1, 1))
    # plt.colorbar()
    # plt.show()