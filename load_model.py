import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet


from glob import glob
from skimage import io
from shutil import copy
from keras.src.models import Model
from keras.src.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.src.applications.inception_resnet_v2 import InceptionResNetV2
from keras.src.layers import Dense, Dropout, Flatten, Input
from keras.src.utils import load_img, img_to_array


if __name__ == '__main__':
    path = './test/h1.jpeg'
    image = load_img(path)  # PIL object
    image = np.array(image, dtype=np.uint8)  # 8 bit array (0,255)
    image1 = load_img(path, target_size=(224, 224))
    image_arr_224 = img_to_array(image1) / 255.0  # Convert into array and get the normalized output

    # Size of the orginal image
    h, w, d = image.shape
    print('Height of the image =', h)
    print('Width of the image =', w)

    fig = px.imshow(image)
    fig.update_layout(width=700, height=500, margin=dict(l=1, r=1, b=1, t=1), xaxis_title='Figure 13 - TEST Image')

    # image_arr_224.shape

    test_arr = image_arr_224.reshape(1, 224, 224, 3)
    # test_arr.shape

    model = tf.keras.models.load_model('./my_model.keras')
    print('Model loaded Sucessfully')

    coords = model.predict(test_arr)
    print(coords)

    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    print(coords)

    coords = coords.astype(np.int32)
    print(coords)

    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    print(pt1, pt2)

    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    fig = px.imshow(image)

    fig.update_layout(width=700, height=500, margin=dict(l=1, r=1, b=1, t=1))
    fig.show()
