import os
import shutil

def copy_files_from_list(txt_file, src_folder, dst_folder):
    # Ensure destination folder exists
    os.makedirs(dst_folder, exist_ok=True)

    with open(txt_file, 'r') as f:
        file_names = [line.strip() for line in f if line.strip()]

    for file_name in file_names:
        src_path = os.path.join(src_folder, file_name)
        dst_path = os.path.join(dst_folder, file_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {file_name}")
        else:
            print(f"Missing: {file_name}")

if __name__ == "__main__":
    # Example usage
    txt_file = "yolo_dataset/test.txt"
    src_folder = "/media/hayden/T7/fireblight/data/rivendale_2-12-2025/image_data/firefly_right"
    dst_folder = "/home/hayden/cmu/kantor_lab/fbanno/yolo_dataset/firefly_right/images/"
    
    copy_files_from_list(txt_file, src_folder, dst_folder)
