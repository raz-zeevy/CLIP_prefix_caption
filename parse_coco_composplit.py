# This lines are meant to change the environment variables
# in order to save the model to a different path
from prepare import prepare
prepare()
#
import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse

SPLIT_NAME = "composplit"

def main(clip_model_type: str):
    """
    Main function for generating embeddings using the CLIP model.

    Args:
        clip_model_type (str): Type of CLIP model to use.

    Returns:
        int: Status code (0) indicating the success of the function.
    """
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/{SPLIT_NAME}_split_{clip_model_name}_train.pkl"
    # Load the CLIP model and the preprocessing function
    clip_model, preprocess = clip.load(clip_model_type, device=device,
                                       jit=False,
   download_root="/cs/snapless/oabend/raz.zeevy/CLIP_prefix_caption/model")
    # Load the captions from a JSON file
    with open('./data/coco/annotations/train_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        # check if the image is in the cosplit train set
        # if not, skip it
        # todo - mentiond above
        d = data[i]
        img_id = d["image_id"]
        filename = f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        # If the image is not found in the train2014 directory, try the val2014 directory
        if not os.path.isfile(filename):
            filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            # Encode the image using the CLIP model
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            # Periodically save the embeddings and captions to a pickle file
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
    # Save the final embeddings and captions to a pickle file
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
