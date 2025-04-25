import json
import os
import random
import argparse

def split_annotations(input_file: str, train_file: str, val_file: str, train_ratio: float = 0.8, seed: int = 42) -> None:
    """
    Splits a COCO annotations file into train and validation JSON files.

    Args:
        input_file (str): Path to the input annotations.json file.
        train_file (str): Path to the output training JSON file.
        val_file (str): Path to the output validation JSON file.
        train_ratio (float): Fraction of the images to allocate for training.
        seed (int): Random seed for reproducibility.
    """
    # Load the full annotations file
    with open(input_file, 'r') as f:
        coco = json.load(f)

    # Extract common parts that are the same for both splits.
    info = coco.get('info', {})
    licenses = coco.get('licenses', [])
    categories = coco.get('categories', [])
    
    # Get all images and annotations.
    images = coco['images']
    annotations = coco['annotations']

    # Shuffle the images for a random split.
    random.seed(seed)
    random.shuffle(images)

    num_images = len(images)
    num_train = int(num_images * train_ratio)
    train_images = images[:num_train]
    val_images = images[num_train:]

    # Build a set of image ids for each subset.
    train_ids = {img['id'] for img in train_images}
    val_ids   = {img['id'] for img in val_images}

    # Filter the annotations for each split based on image_id.
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_ids]
    val_annotations   = [ann for ann in annotations if ann['image_id'] in val_ids]

    # Build the train and validation dicts in COCO format.
    train_data = {
        'info': info,
        'licenses': licenses,
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories
    }
    
    val_data = {
        'info': info,
        'licenses': licenses,
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories
    }

    # Write out the files.
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Created {train_file} with {len(train_images)} images and {len(train_annotations)} annotations.")
    print(f"Created {val_file} with {len(val_images)} images and {len(val_annotations)} annotations.")

def main():
    parser = argparse.ArgumentParser(
        description="Split a COCO annotations.json file into train.json and val.json files."
    )
    parser.add_argument('--input', type=str, default='annotations.json',
                        help="Path to the input annotations.json file.")
    parser.add_argument('--train', type=str, default='train.json',
                        help="Path to the output train JSON file.")
    parser.add_argument('--val', type=str, default='val.json',
                        help="Path to the output validation JSON file.")
    parser.add_argument('--ratio', type=float, default=0.8,
                        help="Fraction of images to assign to the training set (default: 0.8).")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for shuffling (default: 42).")

    args = parser.parse_args()

    # Ensure the input file exists.
    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist.")
        return

    split_annotations(args.input, args.train, args.val, args.ratio, args.seed)

if __name__ == '__main__':
    main()