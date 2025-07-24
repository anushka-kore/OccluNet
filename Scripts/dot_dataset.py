'''Script to create synthetic dataset for OccluNet'''

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
import math

# Configuration
TOTAL_SEQUENCES = 200
TRAIN_RATIO = 0.8
MIN_FRAMES = 3
MAX_FRAMES = 25
IMAGE_SIZE = 1024
DOT_RADIUS = 15
DOT_DIAMETER = DOT_RADIUS * 2
BBOX_SIZE = 40  # 40x40 pixels

# Base directory
BASE_DIR = '/home/akore/mmdet_project/dataset_dot'

# Create directories
os.makedirs(os.path.join(BASE_DIR, 'annotations'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'val'), exist_ok=True)

# Initialize COCO format dictionaries
def init_coco():
    return {
        "images": [],
        "annotations": [],  
        "categories": [{"id": 1, "name": "Dot"}]
    }

train_coco = init_coco()
val_coco = init_coco()

# Generate random dot position that fits within image with margin for dot size
def get_random_dot_position():
    margin = DOT_RADIUS
    x = random.randint(margin, IMAGE_SIZE - margin - 1)
    y = random.randint(margin, IMAGE_SIZE - margin - 1)
    return x, y

# Generate a sequence
def generate_sequence(seq_id, output_dir, coco_data):
    num_frames = random.randint(MIN_FRAMES, MAX_FRAMES)
    center_frame = num_frames // 2
    start_frame = max(1, center_frame - num_frames // 4)
    end_frame = min(num_frames, center_frame + num_frames // 4)
    
    # Get random dot position for this sequence
    dot_x, dot_y = get_random_dot_position()
    # Bounding box coordinates (top-left corner)
    bbox_x = dot_x - DOT_RADIUS
    bbox_y = dot_y - DOT_RADIUS
    
    # Create sequence directory
    seq_dir = os.path.join(BASE_DIR, output_dir, f'sequence_{seq_id}')
    os.makedirs(seq_dir, exist_ok=True)
    
    for frame_num in range(1, num_frames + 1):
        # Create blank grayscale image
        img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color=255)
        draw = ImageDraw.Draw(img)
        
        # Add dot if frame is in the middle 50%
        has_dot = start_frame <= frame_num <= end_frame
        if has_dot:
            # Draw black circle
            draw.ellipse([
                (dot_x - DOT_RADIUS, dot_y - DOT_RADIUS),
                (dot_x + DOT_RADIUS, dot_y + DOT_RADIUS)
            ], fill=0)
        
        # Save image
        frame_path = os.path.join(seq_dir, f'frame_{frame_num:04d}.jpg')
        img.save(frame_path)
        
        # Add to COCO data
        image_id = len(coco_data["images"]) + 1
        rel_path = os.path.join(f'sequence_{seq_id}', f'frame_{frame_num:04d}.jpg')
        coco_data["images"].append({
            "id": image_id,
            "file_name": rel_path,
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "frame_number": frame_num,
            "video_id": seq_id
        })
        
        if has_dot:
            annotation_id = len(coco_data["annotations"]) + 1
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [bbox_x, bbox_y, BBOX_SIZE, BBOX_SIZE],
                "area": BBOX_SIZE*BBOX_SIZE,  # Actual area of the bbox
                "iscrowd": 0
            })

# Generate all sequences
random.seed(42)
np.random.seed(42)

train_count = int(TOTAL_SEQUENCES * TRAIN_RATIO)
val_count = TOTAL_SEQUENCES - train_count

print("Generating training sequences...")
for seq_id in tqdm(range(1, train_count + 1)):
    generate_sequence(seq_id, 'train', train_coco)

print("Generating validation sequences...")
for seq_id in tqdm(range(train_count + 1, train_count + val_count + 1)):
    generate_sequence(seq_id, 'val', val_coco)

# Save COCO annotations
with open(os.path.join(BASE_DIR, 'annotations/train.json'), 'w') as f:
    json.dump(train_coco, f)

with open(os.path.join(BASE_DIR, 'annotations/val.json'), 'w') as f:
    json.dump(val_coco, f)

print("Dataset generation complete!")
print(f"Training sequences: {train_count}")
print(f"Validation sequences: {val_count}")
