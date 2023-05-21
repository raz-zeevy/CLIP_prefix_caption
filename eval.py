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
        default=os.path.expanduser("model/ViT-B-32.pt"),
    )
    parser.add_argument(
        "--image-path",
        help="Path to the image for which captions should be generated",
        default='./Images/COCO_val2014_000000562207.jpg'
    )
    parsed_args = parser.parse_args(args)
    return parsed_args

def eval_using_predict(image_path, model_path):
    from predict_0 import Predictor
    use_beam_search = False
    predictor = Predictor()
    predictor.setup(model_path)
    predictor.predict(image_path, use_beam_search)

def eval_using_load(model_path : str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.jit.load(model_path, device).state_dict()
    print("")

    # image_features = model.encode_image(image_path)
    # captions = model.generate(image_features)
    # print(captions)
    # image = io.imread(image)
    # Convert the image to a PIL image.
    # pil_image = PIL.Image.fromarray(image)
    # image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
    # with torch.no_grad():
    #     prefix = self.clip_model.encode_image(image).to(
    #         self.device, dtype=torch.float32
    #     )
    #     prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)


if __name__ == '__main__':
    parsed_args = check_args(sys.argv[1:])
    # Define the path to the .pt file of the CLIP model
    model_path = parsed_args.model_path

    # Define the path to the image for which captions will be generated
    image_path = parsed_args.image_path

    eval_using_predict(image_path, model_path)
    # eval_using_load(model_path)