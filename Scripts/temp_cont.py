'''
import os
import json
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def load_json(path):
    with open(path) as f:
        return json.load(f)

def bbox_center(bbox):
    x, y, w, h = bbox
    return (x + w/2, y + h/2)

def center_in_box(center, bbox, threshold=1):
    # threshold is in pixels, should be >= half bbox size for your case
    x, y = center
    bx, by, bw, bh = bbox
    # Allow some slack around the bbox
    return (bx - threshold) <= x <= (bx + bw + threshold) and (by - threshold) <= y <= (by + bh + threshold)

def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def plot_confusion_matrix(TP, FP, FN, save_path='confusion_matrix.png'):
    cm = np.array([[TP, FP],
                   [FN, np.nan]])  # TN is blank
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues', vmin=0, interpolation='nearest')
    for i in range(2):
        for j in range(2):
            value = "" if np.isnan(cm[i, j]) else int(cm[i, j])
            ax.text(j, i, value,
                    ha="center", va="center",
                    color="white" if not np.isnan(cm[i, j]) and cm[i, j] > np.nanmax(cm)/2. else "black",
                    fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['GT Pos', 'GT Neg'])
    ax.set_yticklabels(['Pred Pos', 'Pred Neg'])
    ax.set_xlabel('Ground truth')
    ax.set_ylabel('Predicted')
    plt.title('Confusion Matrix')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def find_trajectories(preds_by_frame, frame_ids, center_thresh=15):
    # Each prediction in each frame can start a trajectory
    trajectories = []
    for idx, frame_id in enumerate(frame_ids):
        for pred_idx, pred in enumerate(preds_by_frame.get(frame_id, [])):
            traj_frames = [frame_id]
            traj_preds = [pred]
            traj_scores = [pred['score']]
            last_center = bbox_center(pred['bbox'])
            used = set([(frame_id, pred_idx)])
            # Try to extend this trajectory forward
            for next_idx in range(idx+1, len(frame_ids)):
                next_frame = frame_ids[next_idx]
                found = False
                for next_pred_idx, next_pred in enumerate(preds_by_frame.get(next_frame, [])):
                    next_bbox = next_pred['bbox']
                    if center_in_box(last_center, next_bbox, threshold=center_thresh):
                        if (next_frame, next_pred_idx) not in used:
                            traj_frames.append(next_frame)
                            traj_preds.append(next_pred)
                            traj_scores.append(next_pred['score'])
                            last_center = bbox_center(next_bbox)
                            used.add((next_frame, next_pred_idx))
                            found = True
                            break  # Only one match per frame
                if not found:
                    break
            trajectories.append({
                'frames': traj_frames,
                'bboxes': traj_preds,
                'scores': traj_scores
            })
    return trajectories

def main(gt_annotations_path, pred_annotations_path):
    gt_data = load_json(gt_annotations_path)
    pred_data = load_json(pred_annotations_path)

    # Count unique video_ids in GT
    unique_video_ids = set(img['video_id'] for img in gt_data['images'])
    print(f"Total unique video_ids (sequences) in GT: {len(unique_video_ids)}")

    # Resize factor
    ORIGINAL_SIZE = 1024
    TARGET_SIZE = 640
    SCALE_FACTOR = TARGET_SIZE / ORIGINAL_SIZE

    # Build GT mappings, resizing bboxes
    image_id_to_video_id = {}
    video_id_to_gt_boxes = defaultdict(list)
    video_id_to_frame_ids = defaultdict(list)
    for img in gt_data['images']:
        image_id_to_video_id[img['id']] = img['video_id']
        video_id_to_frame_ids[img['video_id']].append(img['id'])
    for ann in gt_data.get('annotations', []):
        vid = image_id_to_video_id[ann['image_id']]
        x, y, w, h = ann['bbox']
        scaled_bbox = [
            x * SCALE_FACTOR,
            y * SCALE_FACTOR,
            w * SCALE_FACTOR,
            h * SCALE_FACTOR
        ]
        video_id_to_gt_boxes[vid].append(scaled_bbox)

    # Build prediction mappings by frame
    preds_by_frame = defaultdict(list)
    for pred in pred_data:
        preds_by_frame[pred['image_id']].append(pred)

    # Group predictions by sequence
    video_id_to_preds_by_frame = defaultdict(lambda: defaultdict(list))
    for pred in pred_data:
        vid = image_id_to_video_id.get(pred['image_id'])
        if vid is not None:
            video_id_to_preds_by_frame[vid][pred['image_id']].append(pred)

    # Sequence-level evaluation using highest scoring trajectory
    TP, FP, FN = 0, 0, 0
    for vid in unique_video_ids:
        frame_ids = sorted(video_id_to_frame_ids[vid])
        preds_by_frame = video_id_to_preds_by_frame[vid]
        trajectories = find_trajectories(preds_by_frame, frame_ids, center_thresh=15)
        # Score each trajectory
        best_traj = None
        best_score = -np.inf
        for traj in trajectories:
            duration = len(traj['frames'])
            total_score = sum(traj['scores']) * duration
            if total_score > best_score:
                best_score = total_score
                best_traj = traj
        gts = video_id_to_gt_boxes.get(vid, [])
        is_tp = False
        if best_traj is not None and gts:
            for pred in best_traj['bboxes']:
                for gt_bbox in gts:
                    if calculate_iou(pred['bbox'], gt_bbox) > 0.0001:
                        is_tp = True
                        break
                if is_tp:
                    break
        if gts:
            if is_tp:
                TP += 1
            else:
                FN += 1
        else:
            # No GT in this sequence, but predictions exist
            if best_traj is not None:
                FP += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    plot_confusion_matrix(TP, FP, FN)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence-level trajectory evaluation for temporal predictions')
    parser.add_argument('--gt_annotations', required=True, help='Path to ground truth COCO annotations JSON file')
    parser.add_argument('--pred_annotations', required=True, help='Path to predicted annotations JSON file')
    args = parser.parse_args()
    main(
        args.gt_annotations,
        args.pred_annotations
    )
'''

import os
import json
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def load_json(path):
    with open(path) as f:
        return json.load(f)

def bbox_center(bbox):
    x, y, w, h = bbox
    return (x + w/2, y + h/2)

def center_in_box(center, bbox, threshold=1):
    x, y = center
    bx, by, bw, bh = bbox
    return (bx - threshold) <= x <= (bx + bw + threshold) and (by - threshold) <= y <= (by + bh + threshold)

def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def plot_confusion_matrix(TP, FP, FN, save_path='confusion_matrix.png'):
    cm = np.array([[TP, FP],
                   [FN, np.nan]])  # TN is blank
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues', vmin=0, interpolation='nearest')
    for i in range(2):
        for j in range(2):
            value = "" if np.isnan(cm[i, j]) else int(cm[i, j])
            ax.text(j, i, value,
                    ha="center", va="center",
                    color="white" if not np.isnan(cm[i, j]) and cm[i, j] > np.nanmax(cm)/2. else "black",
                    fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['GT Pos', 'GT Neg'])
    ax.set_yticklabels(['Pred Pos', 'Pred Neg'])
    ax.set_xlabel('Ground truth')
    ax.set_ylabel('Predicted')
    plt.title('Confusion Matrix')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def find_trajectories(preds_by_frame, frame_ids, center_thresh=15):
    # Each prediction in each frame can start a trajectory
    trajectories = []
    for idx, frame_id in enumerate(frame_ids):
        for pred_idx, pred in enumerate(preds_by_frame.get(frame_id, [])):
            traj_frames = [frame_id]
            traj_preds = [pred]
            traj_scores = [pred['score']]
            last_center = bbox_center(pred['bbox'])
            used = set([(frame_id, pred_idx)])
            # Try to extend this trajectory forward
            for next_idx in range(idx+1, len(frame_ids)):
                next_frame = frame_ids[next_idx]
                found = False
                for next_pred_idx, next_pred in enumerate(preds_by_frame.get(next_frame, [])):
                    next_bbox = next_pred['bbox']
                    if center_in_box(last_center, next_bbox, threshold=center_thresh):
                        if (next_frame, next_pred_idx) not in used:
                            traj_frames.append(next_frame)
                            traj_preds.append(next_pred)
                            traj_scores.append(next_pred['score'])
                            last_center = bbox_center(next_bbox)
                            used.add((next_frame, next_pred_idx))
                            found = True
                            break  # Only one match per frame
                if not found:
                    break
            trajectories.append({
                'frames': traj_frames,
                'bboxes': traj_preds,
                'scores': traj_scores
            })
    return trajectories

def save_optimized_predictions(optimized_preds, output_json_path):
    with open(output_json_path, 'w') as f:
        json.dump(optimized_preds, f, indent=2)

def visualize_optimized_preds(images_dir, output_dir, optimized_preds, image_id_to_file, gt_image_id_to_bboxes):
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()
    ORIGINAL_SIZE = 1024
    TARGET_SIZE = 640

    # Group optimized preds by image_id for quick lookup
    preds_by_image = defaultdict(list)
    for pred in optimized_preds:
        preds_by_image[pred['image_id']].append(pred)

    for image_id, file_name in image_id_to_file.items():
        input_img_path = os.path.join(images_dir, file_name)
        output_img_path = os.path.join(output_dir, file_name)
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        if not os.path.exists(input_img_path):
            continue
        img = Image.open(input_img_path).convert('RGB')
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(img)

        # Draw GT in red
        for gt_bbox in gt_image_id_to_bboxes.get(image_id, []):
            x, y, w, h = gt_bbox
            draw.rectangle([x, y, x+w, y+h], outline='red', width=2)

        # Draw optimized preds in purple
        for pred in preds_by_image.get(image_id, []):
            x, y, w, h = pred['bbox']
            score = pred['score']
            draw.rectangle([x, y, x+w, y+h], outline='purple', width=2)
            label = f"{score:.2f}"
            try:
                text_bbox = draw.textbbox((x, y), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
            except AttributeError:
                text_width, _ = font.getsize(label)
            label_box = [x, y-20, x+text_width+4, y-2]
            draw.rectangle(label_box, fill='purple')
            draw.text((x+2, y-18), label, fill='white', font=font)
        img.save(output_img_path)

def plot_sequence_level_pr_roc(seq_labels, seq_scores):

    if not seq_scores or not seq_labels or sum(seq_labels) == 0:
        print("No true positives found for sequence-level PR/ROC curve.")
        return

    # PR Curve
    precision, recall, _ = precision_recall_curve(seq_labels, seq_scores)
    plt.figure()
    plt.plot(recall, precision, color='blue', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Sequence-level Precision-Recall Curve')
    plt.grid(True, color='#dddddd')  # Light grey gridlines
    plt.tight_layout()
    plt.savefig('sequence_precision_recall_curve.png')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(seq_labels, seq_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Sequence-level ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, color='#dddddd')  # Light grey gridlines
    plt.tight_layout()
    plt.savefig('sequence_roc_curve.png')
    plt.show()

def main(gt_annotations_path, pred_annotations_path, images_dir, output_json_path, output_dir):
    gt_data = load_json(gt_annotations_path)
    pred_data = load_json(pred_annotations_path)

    # Count unique video_ids in GT
    unique_video_ids = set(img['video_id'] for img in gt_data['images'])
    print(f"Total unique video_ids (sequences) in GT: {len(unique_video_ids)}")

    # Resize factor
    ORIGINAL_SIZE = 1024
    TARGET_SIZE = 640
    SCALE_FACTOR = TARGET_SIZE / ORIGINAL_SIZE

    # Build GT mappings, resizing bboxes
    image_id_to_video_id = {}
    video_id_to_gt_boxes = defaultdict(list)
    video_id_to_frame_ids = defaultdict(list)
    image_id_to_file = {}
    gt_image_id_to_bboxes = defaultdict(list)
    for img in gt_data['images']:
        image_id_to_video_id[img['id']] = img['video_id']
        video_id_to_frame_ids[img['video_id']].append(img['id'])
        image_id_to_file[img['id']] = img['file_name']
    for ann in gt_data.get('annotations', []):
        vid = image_id_to_video_id[ann['image_id']]
        x, y, w, h = ann['bbox']
        scaled_bbox = [
            x * SCALE_FACTOR,
            y * SCALE_FACTOR,
            w * SCALE_FACTOR,
            h * SCALE_FACTOR
        ]
        video_id_to_gt_boxes[vid].append(scaled_bbox)
        gt_image_id_to_bboxes[ann['image_id']].append(scaled_bbox)

    # Build prediction mappings by frame
    preds_by_frame = defaultdict(list)
    for pred in pred_data:
        preds_by_frame[pred['image_id']].append(pred)

    # Group predictions by sequence
    video_id_to_preds_by_frame = defaultdict(lambda: defaultdict(list))
    for pred in pred_data:
        vid = image_id_to_video_id.get(pred['image_id'])
        if vid is not None:
            video_id_to_preds_by_frame[vid][pred['image_id']].append(pred)

    # Sequence-level evaluation using highest scoring trajectory
    TP, FP, FN = 0, 0, 0
    optimized_preds = []
    seq_scores = []
    seq_labels = []
    for vid in unique_video_ids:
        frame_ids = sorted(video_id_to_frame_ids[vid])
        preds_by_frame = video_id_to_preds_by_frame[vid]
        trajectories = find_trajectories(preds_by_frame, frame_ids, center_thresh=1)
        # Score each trajectory
        best_traj = None
        best_score = -np.inf
        for traj in trajectories:
            duration = len(traj['frames'])
            total_score = sum(traj['scores']) * duration
            if total_score > best_score:
                best_score = total_score
                best_traj = traj
        gts = video_id_to_gt_boxes.get(vid, [])
        is_tp = False
        if best_traj is not None and gts:
            for pred in best_traj['bboxes']:
                for gt_bbox in gts:
                    if calculate_iou(pred['bbox'], gt_bbox) > 0.0001:
                        is_tp = True
                        break
                if is_tp:
                    break
        if gts:
            if is_tp:
                TP += 1
                seq_labels.append(1)
                seq_scores.append(best_score)
            else:
                FN += 1
                seq_labels.append(0)
                seq_scores.append(best_score if best_traj is not None else 0)
        else:
            # No GT in this sequence, but predictions exist
            if best_traj is not None:
                FP += 1
                seq_labels.append(0)
                seq_scores.append(best_score)
            # If no best_traj and no GT, skip (no prediction, no GT)

        # Save optimized predictions for this sequence
        if best_traj is not None:
            duration = len(best_traj['frames'])
            total_score = sum(best_traj['scores']) * duration
            for frame_id, pred in zip(best_traj['frames'], best_traj['bboxes']):
                optimized_preds.append({
                    'image_id': frame_id,
                    'bbox': pred['bbox'],
                    'score': total_score,
                    'category_id': pred.get('category_id', 0)
                })

    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Save optimized predictions
    save_optimized_predictions(optimized_preds, output_json_path)

    # Visualize and save images
    visualize_optimized_preds(images_dir, output_dir, optimized_preds, image_id_to_file, gt_image_id_to_bboxes)

    plot_confusion_matrix(TP, FP, FN)

    # Sequence-level PR/ROC curves
    plot_sequence_level_pr_roc(seq_labels, seq_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence-level trajectory evaluation for temporal predictions')
    parser.add_argument('--gt_annotations', required=True, help='Path to ground truth COCO annotations JSON file')
    parser.add_argument('--pred_annotations', required=True, help='Path to predicted annotations JSON file')
    parser.add_argument('--images_dir', required=True, help='Path to validation images directory')
    parser.add_argument('--output_json', required=True, help='Path to output JSON file for optimized predictions')
    parser.add_argument('--output_dir', required=True, help='Path to output directory for visualizations')
    args = parser.parse_args()
    main(
        args.gt_annotations,
        args.pred_annotations,
        args.images_dir,
        args.output_json,
        args.output_dir
    )
