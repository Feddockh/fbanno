import os
import json
import numpy as np
from typing import List, Dict, Tuple
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import Image, BoundingBoxes, BoundingBoxFormat, Mask
from utils.camera import Camera
from utils.visual import plot


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rivendale_dataset")
# DATA_DIR = os.path.join(BASE_DIR, "erwiam_dataset")

class SetType:
    ALL = "annotations"
    TRAIN = "train"
    VAL = "val"

class MultiCamDataset(Dataset):
    def __init__(self, base_dir: str, cameras: List[Camera], 
                 set_type: SetType = SetType.ALL, transforms = None, 
                 undistort_rectify: bool = False):
        """
        Initialize the multi-cam dataset with the base directory and camera names.
        All cameras must have the same number of images.
        """
        self.base_dir = base_dir
        self.cameras = cameras
        self.set_type = set_type
        self.transforms = transforms
        self.annotations: Dict[str, COCO] = {}

        # Load annotations from COCO format for each camera
        for cam in cameras:
            # Check that the camera parameters have been loaded
            if cam.width == 0 or cam.height == 0:
                raise ValueError(f"Camera parameters not loaded for {cam.name}. Call load_params() first.")

            cam_dir = os.path.join(base_dir, cam.name)
            if not os.path.exists(cam_dir):
                raise ValueError(f"Camera directory {cam_dir} does not exist.")
            cam_annotations_path = os.path.join(cam_dir, set_type + '.json')

            # If there is no annotations file, create a new one from cam0
            if not os.path.exists(cam_annotations_path):
                print(f"Annotations file {cam_annotations_path} does not exist... Creating a new one from cam0.")
                cam0_annotations_path = os.path.join(base_dir, cameras[0].name, set_type + '.json')

                # If the cam0 annotations file does not exist, raise an error
                if not os.path.exists(cam0_annotations_path):
                    raise ValueError(f"Annotations file {cam0_annotations_path} does not exist.")
                
                # Copy the cam0 annotations file to the current camera directory (removing annotations, and updating width/height)
                cam_annotations_path = self._coco_annos_copy(cam0_annotations_path, cam_dir, cam)
            self.annotations[cam.name] = COCO(cam_annotations_path)

        # Load the image ids from the first camera (should be the same for all cameras)
        cam0_coco = self.annotations[cameras[0].name]
        self.ids = cam0_coco.getImgIds()

        # Build class list from the first camera
        category_ids = cam0_coco.getCatIds()
        if len(category_ids) == 0:
            raise ValueError(f"No categories found in {cam0_coco.dataset['categories']}")
        categories = cam0_coco.loadCats(category_ids)
        self.class_names = [cat['name'] for cat in categories]
        print(f"Loaded {len(self.class_names)} classes: {self.class_names}")

        self.undistort_rectify = undistort_rectify

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx) -> Dict[str, Tuple[Image, Dict[str, torch.Tensor]]]:
        """
        Get a sample from the dataset. 
        
        This returns a dictionary with the camera name as the key corresponding to 
        each cameras matching (image, annotations) tuple.

        Each camera has its own image and annotations that correspond to the same 
        capture instance. The images are loaded in RGB format and the annotations 
        are loaded in COCO format. The annotations are converted to tv tensors
        so that they can be transformed using the same transforms as the images.

        The annotations are a dictionary with keys 'boxes', 'masks', and 'labels'.
        The 'boxes' are in XYXY format, the 'masks' are in binary format, and the
        'labels' are in integer format.
        """
        sample = {}

        # Get the image id and filename (same for all cameras)
        img_id = self.ids[idx]
        img_filename = self.annotations[self.cameras[0].name].imgs[img_id]['file_name']

        # Iterate through each camera
        for cam in self.cameras:
            
            ## Load the image ##
            # Formulate the full path for the image (file paths should be the same except for the camera name)
            img_path = os.path.join(self.base_dir, cam.name, "images", img_filename)
            # Check if the image file exists
            if not os.path.exists(img_path):
                raise ValueError(f"Image file {img_path} does not exist.")
            # Load in the image as a tensor with dimensions [C, H, W]
            img = decode_image(img_path)
            img = F.to_dtype(img, torch.float32, scale=True)

            ## Load the annotations ##
            # Get the annotations for this image
            ann_ids = self.annotations[cam.name].getAnnIds(imgIds=img_id)
            anns = self.annotations[cam.name].loadAnns(ann_ids)

            # Define the canvas size as (height, width)
            canvas_size = (img.shape[1], img.shape[2])

            # If no annotations exist for this image, create empty targets.
            if len(anns) == 0:
                boxes_tv = BoundingBoxes(torch.empty((0, 4), dtype=torch.float32),
                                        format=BoundingBoxFormat.XYWH,
                                        canvas_size=canvas_size)
                masks_tv = Mask(torch.empty((0, canvas_size[0], canvas_size[1]), dtype=torch.uint8))
                labels_tensor = torch.empty((0,), dtype=torch.int64)
            else:
                ## Format the annotations as tv tensor ##
                # Each annotation dict contains a key "bbox" with [x, y, width, height]
                boxes_list = [ann['bbox'] for ann in anns]
                # Convert the list of boxes to a tensor of shape [num_boxes, 4]
                boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
                # Create the BoundingBoxes TVTensor with the boxes in XYWH format
                boxes_tv = BoundingBoxes(boxes_tensor, format=BoundingBoxFormat.XYWH, canvas_size=canvas_size)
                # Convert the boxes to XYXY format using transforms v2 for tv tensors
                boxes_tv = F.convert_bounding_box_format(boxes_tv, new_format=BoundingBoxFormat.XYXY)

                # Each annotation dict contains a key "segmentation" with lists of polygons
                coco = self.annotations[cam.name]
                # Iterate through each annotation and convert the segmentation to a binary mask
                masks_list = []
                for ann in anns:
                    if ann['segmentation'] != []:
                        masks_list.append(coco.annToMask(ann))
                    else:
                        masks_list.append(np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8))
                # Convert the list to a single numpy array of shape [num_masks, height, width]
                masks_np = np.array(masks_list, dtype=np.uint8)
                # Convert the list of masks to a tensor of shape [num_masks, height, width]
                masks_tensor = torch.tensor(masks_np, dtype=torch.uint8)
                # Create the Mask TVTensor with the masks
                masks_tv = Mask(masks_tensor)

                # Each annotation dict contains a key "category_id" with the class label
                labels_list = [ann['category_id'] for ann in anns]
                # Convert the list of labels to a tensor of shape [num_labels]
                labels_tensor = torch.tensor(labels_list, dtype=torch.int64)

            # Create a dictionary of annotations
            target = {
                'boxes': boxes_tv,
                'masks': masks_tv,
                'labels': labels_tensor
            }

            # # Add grid lines to the image for visualization
            # spacing = img.shape[1] // 10
            # img[:, ::spacing, :] = 1.0; img[:, :, ::spacing] = 1.0

            # If the undistort_rectify flag is set, undistort and rectify the image and annotations
            if self.undistort_rectify:
                img = cam.undistort_rectify_image(img)
                target = cam.undistort_rectify_target(target)

            # Check if the target is a tv tensor (necessary for transforms)
            if not isinstance(target['boxes'], BoundingBoxes) or not isinstance(target['masks'], Mask):
                raise ValueError("Target must be a tv tensor with boxes and masks.")

            # Apply transform if provided (can apply to both image and annotations)
            # due to the use of torchvision.transforms.v2 and torchvision.tv_tensors
            if self.transforms:
                img, target = self.transforms(img, target)

            # Each annotation dict contains a key "ids" with the annotation ID
            ids = torch.tensor(ann_ids, dtype=torch.int64)
            target['ids'] = ids

            # Add the image and annotations to the sample for this camera
            sample[cam.name] = (img, target, img_path)

        # Filter out invalid targets
        sample = self._filter_invalid_targets(sample)
        return sample
    
    def _coco_annos_copy(self, input_json_path: str,
                    output_dir: str,
                    cam: Camera) -> str:
        """
        Copy a COCO-format annotations file, adjust width/height for images
        of a given camera, and remove all annotations.

        Args:
            input_json_path: Path to original COCO annotations.json.
            output_dir:      Directory to write the new annotations.json.
            cam:             Camera object for camera name, width, and height.

        Returns:
            The path to the new annotations file.
        """
        os.makedirs(output_dir, exist_ok=True)

        with open(input_json_path, 'r') as f:
            coco = json.load(f)

        # Adjust width/height on images for this camera
        for img in coco.get("images"):
            img["width"] = cam.width
            img["height"] = cam.height

        # Remove all annotations
        coco["annotations"] = []

        out_path = os.path.join(output_dir, os.path.basename(input_json_path))
        with open(out_path, 'w') as f:
            json.dump(coco, f, indent=2)
        return out_path
    
    def _filter_invalid_targets(self, sample: Dict[str, Tuple[Image, Dict[str, torch.Tensor]]]) \
             -> Dict[str, Tuple[Image, Dict[str, torch.Tensor]]]:
        """
        Remove any annotation whose box has <=0 area in ANY camera
        """
        for cam in self.cameras:
            img, target, img_path = sample[cam.name]
            boxes = target['boxes']
            x1, y1, x2, y2 = boxes.unbind(dim=1)
            keep = (x2 - x1) * (y2 - y1) > 0
            boxes = boxes[keep]
            masks = target['masks'][keep] if target['masks'].shape[0] > 0 else target['masks']
            labels = target['labels'][keep]
            ids = target['ids'][keep]
            sample[cam.name] = (img, {'boxes': boxes, 'masks': masks, 'labels': labels, 'ids': ids}, img_path)

        return sample
    
    def get_class_names(self):
        """
        Get the class names for this dataset.
        """
        return self.class_names

    def save_annos_coco(self, image_id: int, target: Dict[str, torch.Tensor], cam: Camera):
        """
        Append the boxes+labels in `target` for `image_id` into the COCO JSON
        at the cam annotation path. Uses the IDs in `target['ids']` as the annotation
        IDs.

        Args:
            image_id:             the COCO image_id to annotate (1-indexed).
            target:               dict with keys:
                                    - 'boxes' (Tensor[N,4] XYXY),
                                    - 'labels' (Tensor[N], category IDs),
                                    - 'ids' (Tensor[N], annotation IDs to reuse).
            cam:                  the camera object for the current camera.
        """
        # If the cam is the first camera, throw an error
        if cam == self.cameras[0]:
            raise ValueError("Cannot save annotations for the first camera.")
        
        coco: COCO = self.annotations[cam.name]
        dataset_dict = coco.dataset  # Plain Python dict

        # Ensure there's an annotations list
        anns_list = dataset_dict.get('annotations', [])

        # Get the boxes, labels, and ids from the target
        boxes  = target['boxes'].tolist()
        labels = target['labels'].tolist()
        ids    = target['ids'].tolist()

        # Save new annotations
        # TODO: Handle case where boxes are empty
        for box, cat, ann_id in zip(boxes, labels, ids):

            # Convert the bbox from XYXY to XYWH format and compute the area
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            area = w * h

            # If the annotation ID already exists, reuse it
            ann = {
                'id'         : int(ann_id),
                'image_id'   : image_id,
                'category_id': int(cat),
                'segmentation': [],
                'bbox'       : [x1, y1, w, h],
                'area'       : area,
                'iscrowd'    : 0,
            }
            
            # Add the annotation to the list of annotations, or update it if it already exists
            updated = False
            for i, a in enumerate(anns_list):
                if a['id'] == ann_id:
                    anns_list[i] = ann
                    updated = True
                    break
            # If the annotation ID does not exist, add it to the list
            if not updated:
                anns_list.append(ann)

        # Put it back on the dataset dict
        dataset_dict['annotations'] = anns_list

        # Write out the updated JSON
        cam_annotations_path = os.path.join(self.base_dir, cam.name, self.set_type + ".json")
        with open(cam_annotations_path, 'w') as f:
            json.dump(dataset_dict, f, indent=2)
    
def demo():
    # Create the cameras
    cam0 = Camera("firefly_left") # Use this for the rivendale dataset
    cam1 = Camera("ximea") # Use this for the rivendale dataset
    cameras = [cam0, cam1]
    cam0.load_params(), cam1.load_params()

    # Define the transforms
    transforms = v2.Compose([
        # v2.Resize((1024, 1024), antialias=True), # Higher for finer details
        # v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor(),
    ])

    # Create the dataset
    dataset = MultiCamDataset(DATA_DIR, cameras, set_type=SetType.ALL, transforms=transforms)
    view_idx = 170 # Make sure this index is valid for your dataset
    img, target, img_path = dataset[view_idx][cam0.name] 

    # Print the shape of the image and annotations and plot the image with annotations
    print(f"Image shape: {img.shape}")
    print(f"Annotation boxes shape: {target['boxes'].shape}")
    print(f"Annotation masks shape: {target['masks'].shape}")
    print(f"Annotation labels shape: {target['labels'].shape}")
    print(f"Annotation labels: {target['labels']}")
    plot([(img, target)])

    # Show rectified image
    dataset_rect = MultiCamDataset(DATA_DIR, cameras, set_type=SetType.ALL, transforms=transforms, undistort_rectify=True)
    img0_rect, tgt0_rect, _ = dataset_rect[view_idx][cam0.name]
    print(f"Rectified image shape: {img0_rect.shape}")
    plot([[(img, target)], [(img0_rect, tgt0_rect)]], row_title=["Original Image", "Rectified Image"])

    # Show the undistorted/retified image and the re-distorted/un-rectified image
    img0_unrect = cam0.undistort_rectify_image(img0_rect, inverse=True)
    tgt0_unrect = cam0.undistort_rectify_target(tgt0_rect, inverse=True)
    print(f"Unrectified image shape: {img0_unrect.shape}")
    plot([[(img0_rect, tgt0_rect)], [(img0_unrect, tgt0_unrect)], [(img, target)]], row_title=["Rectified Image", "Unrectified Image", "Original Image"])

if __name__ == '__main__':
    demo()