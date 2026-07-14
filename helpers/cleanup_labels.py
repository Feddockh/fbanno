import os
import glob
import argparse

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

def get_base(path):
    return os.path.splitext(os.path.basename(path))[0]

def parse_args():
    p = argparse.ArgumentParser(
        description="Prune images and labels so only matching pairs remain, and rebuild train.txt"
    )
    p.add_argument("--image-dir",    required=True, help="Directory containing images")
    p.add_argument("--label-dir",    required=True, help="Directory containing .txt labels")
    p.add_argument("--train-file",   required=True, help="Output list of valid image paths")
    return p.parse_args()

def main():
    args = parse_args()
    IMAGE_DIR = args.image_dir
    LABEL_DIR = args.label_dir
    TRAIN_FILE = args.train_file

    # Scan all images
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths += glob.glob(os.path.join(IMAGE_DIR, f'*{ext}'))
    image_bases = {get_base(p): p for p in image_paths}

    # Scan all labels
    label_paths = glob.glob(os.path.join(LABEL_DIR, '*.txt'))
    label_bases = {get_base(p): p for p in label_paths}

    # Delete images without labels
    orphan_images = sorted(set(image_bases) - set(label_bases))
    if orphan_images:
        print(f"\n>>> Deleting {len(orphan_images)} images with NO label files:")
        for base in orphan_images:
            img_path = image_bases[base]
            print("  Deleting", img_path)
            os.remove(img_path)  # Uncomment this line to actually delete files
            # also remove from our dict so it won't show up later
            del image_bases[base]
    else:
        print("\nNo images without labels.")

    # Delete labels without images
    orphan_labels = sorted(set(label_bases) - set(image_bases))
    if orphan_labels:
        print(f"\n>>> Deleting {len(orphan_labels)} orphan label files:")
        for base in orphan_labels:
            lp = label_bases[base]
            print("  Deleting", lp)
            os.remove(lp)  # Uncomment this line to actually delete files
            del label_bases[base]
    else:
        print("\nNo orphan label files to delete.")

    # Remaining valid bases (intersection)
    valid_bases = sorted(set(image_bases) & set(label_bases))
    valid_image_paths = [image_bases[b] for b in valid_bases]

    # Rewrite train.txt
    os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)
    with open(TRAIN_FILE, 'w') as f:
        for img in valid_image_paths:
            # Ensure forward slashes in the list file
            f.write(img.replace('\\', '/') + '\n')

    print(f"\nKept {len(valid_image_paths)} image/label pairs.")
    print(f"Wrote {len(valid_image_paths)} lines to {TRAIN_FILE}.")

if __name__ == '__main__':
    main()
