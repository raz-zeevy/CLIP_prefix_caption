'''
this file is a modification of parse_coco.py from the original repo
it creates the embeddings for the train captions for the 4 splits
of the compositional generalization dataset
'''

# These lines are meant to change the environment variables
# in order to save the model to a different path
from prepare import prepare

prepare()
#
from typing import List, Dict
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


def load_splits(dataset_splits_folder: str) -> List[Dict[str, List[int]]]:
    '''
    load the 4 splits into an array of dictionary each contains the split data
    :param dataset_splits_folder:
    :return:
    '''
    splits = []
    for i in range(1, 5):
        split_name = f"dataset_splits_{i}.json"
        split_path = os.path.join(dataset_splits_folder, split_name)
        with open(split_path, 'r') as f:
            split_data = json.load(f)
            for key in split_data:
                if key not in split_data:
                    split_data[key] = []
                # Extend the list of images for each split
                if split_data[key]:
                    split_data[key].extend(split_data[key])
            splits.append(split_data)
    return splits


def check_if_image_in_split(img_id: int, split: dict) -> bool:
    """
    Checks if an image is in the split.

    Args:
        img_id (int): Image ID.
        split (dict): Dictionary containing the split.

    Returns:
        bool: True if the image is in the split, False otherwise.
    """
    return img_id in split["train"] or img_id in split["val"] or img_id in \
        split["test"]


def create_embedding_pkl(clip_model_type: str, split: dict, split_index : int):
    """
        Main function for generating embeddings using the CLIP model.

        Args:
            clip_model_type (str): Type of CLIP model to use.

        Returns:
            int: Status code (0) indicating the success of the function.
            :param type1:
            :param split:
        """
    print(f"creating embedding for split_{split_index}")
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/{SPLIT_NAME}_split_{split_index}" \
               f"_{clip_model_name}_train.pkl"
    # Load the CLIP model and the preprocessing function
    clip_model, preprocess = clip.load(clip_model_type, device=device,
                                       jit=False,
                                       download_root="/cs/snapless/oabend/raz.zeevy/CLIP_prefix_caption/model"
                                       )
    # Load the captions from a JSON file
    with open('./data/coco/annotations/train_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        # check if the image is in the cosplit train set
        if not check_if_image_in_split(img_id, split): continue
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
                pickle.dump(
                    {"clip_embedding": torch.cat(all_embeddings, dim=0),
                     "captions": all_captions}, f)
    # Save the final embeddings and captions to a pickle file
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0),
                     "captions": all_captions}, f)
    print(f'Done split {split_index}')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


def main(clip_model_type: str, dataset_splits_folder: str, split_index : int):
    splits = load_splits(dataset_splits_folder)
    if split_index is not None:
        create_embedding_pkl(clip_model_type, splits[split_index-1],
                             split_index)
        return
    for i, split in enumerate(splits):
        create_embedding_pkl(clip_model_type, split, i+1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32",
                        choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--dataset-splits', default="./dataset_splits")
    parser.add_argument('--index', default=None)
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.dataset_splits, int(args.index)))
