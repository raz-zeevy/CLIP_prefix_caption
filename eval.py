import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from torch import nn
import argparse
import sys, os
from predict import *

def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        help="Path to the trained model",
        default=os.path.expanduser("model/ViT-B-32.pt"),
    )
    parser.add_argument(
        "--image-path",
        help="Path to the image for which captions should be generated",
        default='./Images/COCO_val2014_000000562207.jpg'
    )
    parsed_args = parser.parse_args(args)
    return parsed_args

if __name__ == '__main__':
    parsed_args = check_args(sys.argv[1:])
    # Define the path to the .pt file of the CLIP model
    model_path = parsed_args.model_path

    # Define the path to the image for which captions will be generated
    image_path = parsed_args.image_path
    # use_beam_search = False
    # predictor = Predictor()
    # predictor.setup()
    # predictor.predict(image_path, model_path, use_beam_search)
    model = torch.jit.load(model_path)
    image_features = model.encode_image(image_path)
    captions = model.generate(image_features)
    print(captions)
