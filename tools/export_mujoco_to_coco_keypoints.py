import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import importlib

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

# Define a function to convert ViCoSClothDataset annotations to COCO format
def convert_to_coco_format(subfolders, output_file, R = 25):
    CAT_CENTER_ID = 1
    CAT_CENTER_NAME = "corner"

    # Create COCO format dictionary
    coco_format = {
        "info": {                
            "version": "v1", 
            "description": "RTFM MUJOCO Dataset",
            "contributor": "domen.tabernik@fri.uni-lj.si", 
            "url": "", 
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "categories": [{"supercategory": CAT_CENTER_NAME, "id": CAT_CENTER_ID, "name": CAT_CENTER_NAME, "keypoints": ["center"],}],
        "licenses": [],
        "images": [],
        "annotations": [],        
    }

    # Load annotations
    from datasets.MuJoCoDataset import RTFMDataset
    db = RTFMDataset(root_dir=ROOT_DIR, subfolder=subfolders, transform=None, use_depth=False, correct_depth_rotation=False, use_normals=False)
    
    annt_id = 0
    # Loop through images in dataset    
    for img_id,item in enumerate(tqdm(db)):
        filename = item['im_name']
        
        # change filename relative to output_file
        filename = filename.replace(os.path.dirname(output_file),".")        
        
        # Add image information to COCO format dictionary
        coco_format["images"].append({
            "id": img_id,
            "file_name": filename,
            "height": item['im_size'][1],
            "width": item['im_size'][0]
        })
        
        # Loop through centers for current image
        gt_centers = item['center']
        gt_centers = gt_centers[(gt_centers[:, 0] > 0) | (gt_centers[:, 1] > 0), :]

        for center in gt_centers:
            # calculate bounding box in XYWH format by adding R to center
            bbox = [center[0]-R, center[1]-R, 2*R, 2*R]

            # Add annotation information to COCO format dictionary
            coco_format["annotations"].append({
                "id": annt_id,
                "image_id": img_id,
                "category_id": CAT_CENTER_ID,                
                "bbox": bbox,
                "segmentation": [[bbox[0], bbox[1],  
                                 bbox[0]+bbox[2], bbox[1], 
                                 bbox[0]+bbox[2], bbox[1]+bbox[3], 
                                 bbox[0], bbox[1]+bbox[3],
                                 bbox[0], bbox[1]]],
                "keypoints": [center[0], center[1], 2],
                "num_keypoints": 1,
                "area": int(np.prod(bbox[2:])),
                "iscrowd": 0
            })
            annt_id+=1
    
    # Save COCO format dictionary to output directory
    with open(output_file, "w") as f:
        json.dump(coco_format, f)


if __name__ == "__main__":
    ROOT_DIR = '/storage/datasets/ClothDataset/'
    
    mujoco_subfolders = ['mujoco',
                         'mujoco_all_combinations_normal_color_temp',
                         'mujoco_all_combinations_rgb_light',
                         'mujoco_white_desk_HS_extreme_color_temp',
                         'mujoco_white_desk_HS_normal_color_temp']
    
    convert_to_coco_format(mujoco_subfolders, output_file=os.path.join(ROOT_DIR, 'mujoco_all_train_coco_format_with_keypoints.json'))