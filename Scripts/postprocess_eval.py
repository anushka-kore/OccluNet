'''Postprocess OccluNet validation results to filter out the highest confidence prediction per DSA sequence for further evaluation through temporal consistency'''

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

def deduplicate_gts(gts):
    seen = set()
    unique_gts = []
    for gt in gts:
        key = (gt['image_id'], tuple(np.round(gt['bbox'], 2)))
        if key not in seen:
            seen.add(key)
            unique_gts.append(gt)
    return unique_gts

def get_highest_score_preds(preds):
    best_preds = {}
    for pred in preds:
        img_id = pred['image_id']
        if img_id not in best_preds or pred['score'] > best_preds[img_id]['score']:
            best_preds[img_id] = pred
    return list(best_preds.values())

def bbox_center(bbox):
    return (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def match_preds_to_gts(preds, gts, iou_thr=0.0001):
    gt_by_img = {}
    for gt in gts:
        gt_by_img.setdefault(gt['image_id'], []).append(gt)
    TP, FP, FN = 0, 0, 0
    distances = []
    matched_gt_ids = set()
    tp_details = []
    fp_details = []
    for pred in preds:
        img_id = pred['image_id']
        pred_bbox = pred['bbox']
        gts_img = gt_by_img.get(img_id, [])
        best_iou = 0
        best_gt = None
        for gt in gts_img:
            iou_val = iou(pred_bbox, gt['bbox'])
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt = gt
        if best_iou >= iou_thr and best_gt and best_gt['id'] not in matched_gt_ids:
            TP += 1
            matched_gt_ids.add(best_gt['id'])
            pred_center = bbox_center(pred_bbox)
            gt_center = bbox_center(best_gt['bbox'])
            dist = np.linalg.norm(np.array(pred_center) - np.array(gt_center))
            distances.append(dist)
            tp_details.append({
                'image_id': img_id,
                'pred_bbox': pred_bbox,
                'gt_bbox': best_gt['bbox'],
                'pred_center': pred_center,
                'gt_center': gt_center,
                'distance': dist
            })
        else:
            FP += 1
            pred_center = bbox_center(pred_bbox)
            # Find nearest GT center (if any GTs in this image)
            min_dist = None
            if gts_img:
                gt_centers = [bbox_center(gt['bbox']) for gt in gts_img]
                dists = [np.linalg.norm(np.array(pred_center) - np.array(gt_c)) for gt_c in gt_centers]
                min_dist = min(dists)
            else:
                min_dist = None  # Or set to np.nan if no GTs in image
            fp_details.append({
                'image_id': img_id,
                'pred_bbox': pred_bbox,
                'pred_center': pred_center,
                'distance': min_dist
            })
    total_gt = len(gts)
    FN = total_gt - len(matched_gt_ids)
    return TP, FP, FN, distances, tp_details, fp_details

def plot_confusion_matrix(TP, FP, FN, save_path='confusion_matrix.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array([[TP, FP],
                   [FN, np.nan]])  # TN is blank

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues', vmin=0, interpolation='nearest')

    # Show numbers, but blank for nan
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

def plot_distances(distances):
    plt.figure(figsize=(8,4))
    plt.hist(distances, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distances between centers of TP predictions and GT')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tp_center_distances.png')
    plt.show()

def visualize_tp_distance(tp_details, stat='min', save_prefix='tp_distance'):
    if not tp_details:
        print("No TP details to visualize.")
        return
    if stat == 'min':
        idx = np.argmin([d['distance'] for d in tp_details])
        label = 'Min'
    elif stat == 'max':
        idx = np.argmax([d['distance'] for d in tp_details])
        label = 'Max'
    elif stat == 'avg':
        avg_dist = np.mean([d['distance'] for d in tp_details])
        idx = np.argmin([abs(d['distance']-avg_dist) for d in tp_details])
        label = 'Avg'
    else:
        return
    d = tp_details[idx]
    fig, ax = plt.subplots()
    # Draw GT bbox
    gt_rect = patches.Rectangle((d['gt_bbox'][0], d['gt_bbox'][1]), d['gt_bbox'][2], d['gt_bbox'][3],
                               linewidth=2, edgecolor='r', facecolor='none', label='GT')
    ax.add_patch(gt_rect)
    # Draw Pred bbox
    pred_rect = patches.Rectangle((d['pred_bbox'][0], d['pred_bbox'][1]), d['pred_bbox'][2], d['pred_bbox'][3],
                                 linewidth=2, edgecolor='g', facecolor='none', label='Prediction')
    ax.add_patch(pred_rect)
    # Draw centers
    ax.plot(*d['gt_center'], 'go', label='GT Center')
    ax.plot(*d['pred_center'], 'ro', label='Pred Center')
    # Draw line between centers
    ax.plot([d['gt_center'][0], d['pred_center'][0]], [d['gt_center'][1], d['pred_center'][1]], 'b--', label='Distance')
    ax.legend()
    ax.set_title(f'{label} Distance TP Example\nDistance: {d["distance"]:.2f} px (Image ID: {d["image_id"]})')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_{label.lower()}.png')
    plt.show()

def plot_sorted_scatter_with_cutoff(tp_details, fp_details, cutoff=None):
    """
    Plots a sorted scatter graph of distances for TP and FP, with an optional cutoff line.
    """
    # Combine all details and label them
    all_details = []
    for d in tp_details:
        all_details.append({'distance': d['distance'], 'type': 'TP'})
    for d in fp_details:
        # Only include FPs with valid distance (not None)
        if d['distance'] is not None:
            all_details.append({'distance': d['distance'], 'type': 'FP'})
    # Sort by distance
    all_details.sort(key=lambda x: x['distance'])
    # Prepare data for plotting
    distances = [d['distance'] for d in all_details]
    colors = ['g' if d['type'] == 'TP' else 'r' for d in all_details]
    labels = [d['type'] for d in all_details]
    # Plot
    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(range(len(distances)), distances, c=colors, label='Distance')
    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='TP', markerfacecolor='g', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='FP', markerfacecolor='r', markersize=8)
    ]
    plt.legend(handles=legend_elements)
    plt.xlabel('Sorted Instance Index')
    plt.ylabel('Distance (pixels)')
    plt.title('Sorted Distances between Prediction and GT Centers (TP & FP)')
    # Draw cutoff line if provided
    if cutoff is not None:
        plt.axhline(y=cutoff, color='b', linestyle='--', label=f'Cutoff = {cutoff}')
        plt.legend()
    plt.tight_layout()
    plt.savefig('sorted_scatter_tp_fp_distances.png')
    plt.show()

def main(pred_json, gt_json):
    preds = load_json(pred_json)
    gts = load_json(gt_json)
    if 'annotations' in gts:
        gts = gts['annotations']
    if isinstance(preds, dict) and 'annotations' in preds:
        preds = preds['annotations']
    # Deduplicate GTs
    gts = deduplicate_gts(gts)
    # Keep only highest score per image_id
    preds_best = get_highest_score_preds(preds)
    # Save new bbox.json
    save_json(preds_best, 'filtered_bbox.json')
    print("Filtered predictions saved to filtered_bbox.json")
    # Confusion matrix
    TP, FP, FN, distances, tp_details, fp_details = match_preds_to_gts(preds_best, gts)
    print(f"Confusion Matrix:\nTP: {TP}\nFP: {FP}\nFN: {FN}")
    plot_confusion_matrix(TP, FP, FN)
    # Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    # Distance stats and plot
    if distances:
        print(f"Distance stats for TP:")
        print(f"  Min: {np.min(distances):.2f}")
        print(f"  Max: {np.max(distances):.2f}")
        print(f"  Avg: {np.mean(distances):.2f}")
        plot_distances(distances)
        visualize_tp_distance(tp_details, 'min')
        visualize_tp_distance(tp_details, 'max')
        visualize_tp_distance(tp_details, 'avg')
        plot_sorted_scatter_with_cutoff(tp_details, fp_details, cutoff=30)
    else:
        print("No true positives to compute distances.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python postprocess_and_eval.py <pred_bbox.json> <gt.json>")
    else:
        main(sys.argv[1], sys.argv[2])
