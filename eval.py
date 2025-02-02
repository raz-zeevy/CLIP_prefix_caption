'''
This file is used to evaluate the model on a single image.
'''
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from torch import nn
import argparse
import sys, os
# from predict import *

def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        help="Path to the trained model",
        default=os.path.expanduser("coco_train/coco_prefix_latest.pt"),
    )
    parser.add_argument(
        "--image-path",
        help="Path to the image for which captions should be generated",
        default='./Images/COCO_val2014_000000562207.jpg'
    )
    parsed_args = parser.parse_args(args)
    return parsed_args

def eval_using_predict(image_path, model_path):
    from predict import Predictor
    use_beam_search = False
    predictor = Predictor()
    predictor.setup(model_path)
    print(predictor.predict(image_path, "coco", use_beam_search))

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parsed_args = check_args(sys.argv[1:])
    # Define the path to the .pt file of the CLIP model
    model_path = parsed_args.model_path

    # Define the path to the image for which captions will be generated
    image_path = parsed_args.image_path

    eval_using_predict(image_path, model_path)
    # eval_using_load(model_path)