'''Calculate per-class occlusion detection results for OccluNet'''

import pandas as pd
import ast
from collections import defaultdict

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    # Parse bounding boxes
    x1_gt, y1_gt, w_gt, h_gt = box1
    x2_gt, y2_gt = x1_gt + w_gt, y1_gt + h_gt
    
    x1_pred, y1_pred, w_pred, h_pred = box2
    x2_pred, y2_pred = x1_pred + w_pred, y1_pred + h_pred
    
    # Calculate intersection coordinates
    x_left = max(x1_gt, x1_pred)
    y_top = max(y1_gt, y1_pred)
    x_right = min(x2_gt, x2_pred)
    y_bottom = min(y2_gt, y2_pred)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    gt_area = w_gt * h_gt
    pred_area = w_pred * h_pred
    union_area = gt_area + pred_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def process_files(metadata_path, model_path, iou_threshold=0.0001):  # 0.01% threshold
    # Read CSV files
    metadata_df = pd.read_csv(metadata_path)
    model_df = pd.read_csv(model_path)
    
    # Process metadata.csv
    metadata_df['Filename'] = metadata_df['Filename'].str.replace('_minIP.png', '')
    
    # Merge occlusion location into model.csv
    merged_df = model_df.merge(
        metadata_df[['Filename', 'Occlusion Location']], 
        on='Filename', 
        how='left'
    )
    
    # Initialize dictionaries to store counts
    tp_counts = defaultdict(int)
    fp_counts = defaultdict(int)
    fn_counts = defaultdict(int)
    instance_counts = defaultdict(int)
    
    # Process each row in the merged dataframe
    for _, row in merged_df.iterrows():
        occlusion_loc = row['Occlusion Location']
        if pd.isna(occlusion_loc):
            occlusion_loc = 'Unknown'
        
        instance_counts[occlusion_loc] += 1
        
        try:
            bbox_gt = ast.literal_eval(row['bbox_gt'])
            bbox_pred = ast.literal_eval(row['bbox_modelB'])
        except:
            # Skip rows with invalid bbox format
            continue
        
        # Check if ground truth is empty (no bbox)
        if not bbox_gt or all(v == 0 for v in bbox_gt):
            if bbox_pred and not all(v == 0 for v in bbox_pred):
                fp_counts[occlusion_loc] += 1
        else:
            if not bbox_pred or all(v == 0 for v in bbox_pred):
                fn_counts[occlusion_loc] += 1
            else:
                iou = calculate_iou(bbox_gt, bbox_pred)
                if iou >= iou_threshold:
                    tp_counts[occlusion_loc] += 1
                else:
                    fp_counts[occlusion_loc] += 1
                    fn_counts[occlusion_loc] += 1
    
    # Calculate precision and recall for each occlusion location
    results = []
    all_locations = set(tp_counts.keys()).union(fp_counts.keys()).union(fn_counts.keys()).union(instance_counts.keys())
    
    for loc in sorted(all_locations):
        tp = tp_counts.get(loc, 0)
        fp = fp_counts.get(loc, 0)
        fn = fn_counts.get(loc, 0)
        instances = instance_counts.get(loc, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results.append({
            'Occlusion Location': loc,
            'Instances': instances,
            'Precision': precision,
            'Recall': recall,
            'TP': tp,
            'FP': fp,
            'FN': fn
        })
    
    # Calculate overall metrics
    total_tp = sum(tp_counts.values())
    total_fp = sum(fp_counts.values())
    total_fn = sum(fn_counts.values())
    total_instances = sum(instance_counts.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    results.append({
        'Occlusion Location': 'ALL',
        'Instances': total_instances,
        'Precision': overall_precision,
        'Recall': overall_recall,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn
    })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return merged_df, results_df

if __name__ == "__main__":
    # Example usage
    metadata_file = "/home/akore/metadata_dsa_MIP.csv"
    model_file = "/home/akore/model_b_gt_comparison_OCN1.csv"
    
    merged_df, results_df = process_files(metadata_file, model_file)
    
    # Save the merged dataframe if needed
    merged_df.to_csv("merged_model_with_occlusion.csv", index=False)
    
    # Display the results with instance counts
    print("\nEvaluation Results:")
    print(results_df[['Occlusion Location', 'Instances', 'Precision', 'Recall']].to_string(index=False))
    
    # Optional: Save results to CSV
    results_df.to_csv("evaluation_results.csv", index=False)
