import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from torch import nn
import argparse
import sys, os

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


    # Define the necessary transformations for image preprocessing
    image_transform = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load the CLIP model from the .pt file
    # model = torch.load(model_path, map_location=device)[
    #     'model']
    # model = torch.jit.load(model_path, map_location=device)
    from predict_0 import ClipCaptionModel
    prefix_length = 10
    model = ClipCaptionModel(prefix_length)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set the model to evaluation mode
    model = model.eval()

    # Load and preprocess the image
    image = Image.open(image_path)
    image = image_transform(image).unsqueeze(0)

    # Perform forward pass through the model to get image features
    with torch.no_grad():
        image_features = model.encode_image(image)

    # Generate captions for the image
    captions = model.decode(image_features)

    # Print the generated captions
    for caption in captions:
        print(caption)