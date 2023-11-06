# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:

from __future__ import \
    absolute_import, print_function, division, unicode_literals

import numpy as np
np.random.seed(42)
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import innvestigate.backend as kutils
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
import innvestigate.backend as ibackend
import keras.applications.inception_v3 as inception
import innvestigate.backend.graph as kgraph
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import argparse
import pickle

import tensorflow as tf
# import tensorflow_datasets as tfds


# @tf.function
# def forward_relevance_propagation(model, example_image, analysis):
#     """
#     Propagates relevance through the layers of the model.
#     """
#     new_relevance_j = tf.cast(tf.convert_to_tensor(analysis[0]), tf.float32)

#     return new_relevance_j  # Reverse the list to match the layer order


def compute_jacobian(activations, model, layer_index, R_k):
    """
    Compute the scaled Jacobian matrix for a specific layer.
    """
    # keep_positives = keras.layers.Lambda(lambda x: x * K.cast(K.greater(x, 0), K.floatx()))
    # activations_pos = kutils.apply(keep_positives, activations)
    
    
    prepare_div = keras.layers.Lambda(
            lambda x: x + (K.cast(K.greater_equal(x, 0), K.floatx()) * 2 - 1) * 1e-7)
    

    with tf.GradientTape() as tape:
        tape.watch(activations)
        # layer = kgraph.copy_layer_wo_activation(
        #      model.layers[layer_index], keep_bias=True, name_template="reversed_kernel_%s")
        # Forward pass to compute the intermediate activations
        intermediate_output = model.layers[layer_index](activations)
        # intermediate_output =  kutils.apply(layer, [activations])
    
    jacobian = tape.jacobian(intermediate_output, activations)
    intermediate_output_eps = prepare_div(intermediate_output)
    sk = tf.divide(R_k, intermediate_output_eps)
    # Compute the Jacobian
    elements_list = list(sk.shape)
    for i in range(len(jacobian.shape) - len(elements_list)):
        elements_list.append(1)

    sk_reshaped = tf.reshape(sk, elements_list)
    jacobian_scaled = tf.multiply(jacobian, sk_reshaped)
    return intermediate_output, jacobian_scaled, jacobian, intermediate_output_eps




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

    # model_wo_softmax = innvestigate.model_wo_softmax(model)

    # Create analyzer
    analyzer = innvestigate.create_analyzer("lrp.epsilon", model)

    # Apply analyzer w.r.t. maximum activated output-neuron
    analysis = analyzer.analyze(example_image) 

    # Visualize the result

    num_layers = len(model.layers)
    R_prime = analysis[1:]  # Initial relevance

    # percentile_95 = np.percentile(analysis[1:], 95)

    # # Select elements that are at or above the 95th percentile
    # top_95_percent = arr[arr >= percentile_95]

    # P_jk = analysis[0]
    
    new_relevance_j = tf.cast(tf.convert_to_tensor(analysis[0]), tf.float32)
    J_kj_list = []
    a_j = tf.cast(tf.convert_to_tensor(example_image), tf.float32)

    for j in range(num_layers):  # Iterate backwards through layers
        R_j = tf.convert_to_tensor(analysis[-(j+1)])
        print(R_j)
        R_k = tf.convert_to_tensor(analysis[-(j+2)])
        print(R_k)
        # Compute scaled Jacobian matrix

        a_k, J_kj, J_orig, a_k_eps = compute_jacobian(a_j, model, j, R_k)
       
        perm_from_kj_to_jk =[]
        perm_from_jk_to_kj =[]
        size_sk = len(R_k.shape)     
        size_Rj = len(R_j.shape)     
        
        for i in range(len(R_j.shape)):
            index = size_sk+ i
            perm_from_kj_to_jk.append(index)  

        for i in range(len(R_k.shape)):
            index = size_Rj+ i
            perm_from_kj_to_jk.append(i) 
            perm_from_jk_to_kj.append(index)  

        for i in range(len(R_j.shape)):   
            perm_from_jk_to_kj.append(i)   

        
        # # A_jk =  tf.math.multiply(a_j, tf.ones_like(J_kj))
        # A_jk = tf.math.multiply(a_j, tf.ones_like(R_k))
        # A_jk =  tf.transpose(A_kj, perm_from_kj_to_jk)

        elements_list = list(a_j.shape)

        for i in range(len(J_kj.shape) - len(elements_list)):
            elements_list.append(1)

        a_j_reshaped = tf.reshape(a_j, elements_list)

        # # Step 3: Element-wise product and division
        R_jk = tf.math.multiply(a_j_reshaped, tf.transpose(J_kj, perm_from_kj_to_jk))

        
        # J_kj_list.append(tf.tensordot(tf.transpose(R_jk, perm_from_jk_to_kj), tf.ones_like(new_relevance_j[-1]), int(len(R_j.shape))))

        elements_list = list(R_j.shape)

        for i in range(len(R_jk.shape) - len(elements_list)):
            elements_list.append(1)

        R_j_reshaped = tf.reshape(R_j, elements_list)
        P_jk = tf.math.divide_no_nan(R_jk, R_j_reshaped)

        axis=[]
        for i in range(len(R_k.shape)):   
            axis.append(-(i+1))

        new_relevance_j_reshaped = tf.reshape(new_relevance_j, elements_list)    
        new_R_jk = tf.math.multiply(P_jk, new_relevance_j_reshaped)


        axis=[]
        for i in range(len(R_j.shape)):   
            axis.append(i)

        new_relevance_k = tf.reduce_sum(new_R_jk, axis=axis)

        old_relevance_k = tf.reduce_sum(R_jk, axis=axis)



        P_k = tf.math.divide_no_nan(new_relevance_k, old_relevance_k)






        # J_kj_list.append(P_jk)
        
        # Step 4: Compute new relevances

        
        # new_relevance_jk = tf.math.multiply(P_jk, tf.transpose(J_kj, perm_from_kj_to_jk))

        # new_relevance_jk_prime = tf.math.divide_no_nan(new_relevance_jk, tf.transpose(J_orig, perm_from_kj_to_jk))

        # elements_list = list(R_k.shape)

        # for i in range(len(R_jk.shape) - len(elements_list)):
        #     elements_list.insert(0, 1)

        # a_k_reshaped = tf.reshape(a_k, elements_list)

        # axis=[]
        # for i in range(len(R_k.shape)):   
        #     axis.append(-(i+1))
        # J_kj_list.append(tf.reduce_sum(new_relevance_jk, axis=axis))

        new_relevance_k = tf.math.multiply(R_k, P_k)
        J_kj_list.append(new_relevance_k)
        a_j = a_k
        new_relevance_j = new_relevance_k

    pendulum_model = kmodels.Model(
            inputs=model.inputs,
            outputs=J_kj_list,
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