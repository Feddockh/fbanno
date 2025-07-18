import os

# === CONFIG ===
folder = "/home/hayden/cmu/kantor_lab/fbanno/yolo_dataset/ximea/labels"
image_list_file = "/home/hayden/cmu/kantor_lab/fbanno/yolo_dataset/train.txt"

# === LOAD IMAGE NAMES ===
with open(image_list_file, 'r') as f:
    image_filenames = [line.strip() for line in f if line.strip()]

image_idx = 0

# Loop through 1.txt to 571.txt
for i in range(1, 572):  # 1 to 571 inclusive
    txt_filename = f"{i}.txt"
    txt_path = os.path.join(folder, txt_filename)

    if os.path.exists(txt_path):
        if image_idx >= len(image_filenames):
            print(f"No more image names to use. Stopped at {i}.")
            break

        image_name = image_filenames[image_idx]
        new_txt_name = os.path.splitext(image_name)[0] + ".txt"
        new_txt_path = os.path.join(folder, new_txt_name)

        os.rename(txt_path, new_txt_path)
        print(f"Renamed {txt_filename} → {new_txt_name}")
        
        image_idx += 1  # only consume a name if we used it
    else:
        # File does not exist, skip the name
        image_idx += 1
