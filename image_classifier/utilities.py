#standard package imports
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import json

def process_image(image_path):
    """
    Function to process image for preditions by tensorflow models. 
    Takes input image_path and returns the image object with
    size (1,244,244,3).
    
    INPUT
    image_path: Path to image file
    
    RETURNS
    image: image object, adjusted for tensorflow model processing.
    """
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel("ERROR")
    
    image_size=224
    
    im = Image.open(image_path)
    im = np.asarray(im)
    im = np.expand_dims(im, axis=0)
    im = tf.image.resize(im, (image_size, image_size))
    im /= 255
    im = im.numpy()
    return im

def load_model(model_path):
    """
    Loads tensorflow model.
    
    INPUT
    model_path: path to saved tensorflow model
    
    RETURNS
    model: model object
    """
    model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    return model

def open_json(json_path):
    """
    Open json file and returns dictionary with contents of json file
    
    INPUT
    json_path: Path to json file
    
    RETURNS
    json_dict: Dictionary with contents of json file
    
    
    """
    with open('label_map.json', 'r') as f:
        json_dict = json.load(f)
    return json_dict