#standard package import
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import os

#custom package imports
from utilities import process_image, load_model, open_json

#suppressing tensorflow warnings and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def predict(image_path, model_path, top_k = 5, category_names = None):
    """
    Returns a set prediction of what flower the given input image is, 
    ordered descending on predicted probability.
    
    INPUTS
    image_path: path to image file
    model_path: path to saved prediction model. Assumes tensorflow model
    top_k: top K predictions to return
    category_names: Path to json mapping class numbers to flower names. If not specified, 
    the returned output only includes class numbers. 
    
    RETURNS
    classes: Classes of predicted flowers in descending order based on probabilities.
    Will be of integers if `category_names`is unspecified. Else, includes names of 
    predicted flowers. 
    probs: Proabilities of predicted classes in descending order. 
    """
    #process image from image_path
    image = process_image(image_path)
   
    #load the model from model_path
    model = load_model(model_path)
    
    #make predictions on the image with the model
    ps = model.predict(image)
    
    #sort indices from predictions descending based on probabilities
    psi = np.argsort(-ps)
    
    #get top K probabilities from classes
    classes = psi[0][:top_k]
    
    #use classes as indices to find the top K predictions
    probs = np.take(ps,classes)
    
    #adding 1 to index to get correct class values
    classes += 1
    
    #check if category names are specified and translate classes to class_names.
    if(category_names):
        class_names = open_json(category_names)
        class_str = [str(x) for x in classes]
        classes = [class_names.get(x) for x in class_str]
        
    return classes, probs

def arg_parser():
    #setting up argument parser
    parser = argparse.ArgumentParser(description='Predict flower from image')
    
    #create arguments: image, model, top_k, and category_names
    parser.add_argument("image", type=str, help="path to image file")
    parser.add_argument("model", type=str, help="path to model file")
    parser.add_argument("--top_k", type=int, default=5, nargs='?', help="number, K, top predictions to be returned")
    parser.add_argument("--category_names", type=str, nargs='?', help="path to json file containing categories")
    
    #parse arguments to be returned
    args = parser.parse_args()
    
    return args

def main():
    #parse arguments
    args = arg_parser()
    
    #predict
    classes, probs = predict(args.image, args.model, args.top_k, args.category_names)
    
    #print predictions and classes
    print(classes)
    print(probs)
    

if __name__ == "__main__":
    main()
