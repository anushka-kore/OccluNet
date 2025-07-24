'''McNemar's test for model comparison'''

import pandas as pd
import numpy as np
import ast
from statsmodels.stats.contingency_tables import mcnemar

def iou(boxA, boxB):
    # box: [x, y, w, h]
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB

    # Intersection
    x_left = max(xA, xB)
    y_top = max(yA, yB)
    x_right = min(xA + wA, xB + wB)
    y_bottom = min(yA + hA, yB + hB)
    inter_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    areaA = wA * hA
    areaB = wB * hB
    union_area = areaA + areaB - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / (union_area + 1e-6)

def is_correct(pred_bbox, gt_bbox, iou_thr=0.5):
    if pred_bbox is None or gt_bbox is None:
        return False
    return iou(pred_bbox, gt_bbox) >= iou_thr

def parse_bbox(bbox_str):
    try:
        bbox = ast.literal_eval(bbox_str)
        if isinstance(bbox, list) and len(bbox) == 4:
            return bbox
    except Exception:
        pass
    return None

# Load CSVs
csv1 = pd.read_csv('/home/akore/model_b_gt_comparison_OCN1.csv')
csv2 = pd.read_csv('/home/akore/model_b_gt_comparison_OCN2.csv')

# Merge on Filename
merged = pd.merge(csv1, csv2, on='Filename', suffixes=('_1', '_2'))

print(f"Number of matched Filename values: {len(merged)}")
results = []

for idx, row in merged.iterrows():
    bbox_pred1 = parse_bbox(row['bbox_modelB_1'])
    bbox_pred2 = parse_bbox(row['bbox_modelB_2'])
    bbox_gt = parse_bbox(row['bbox_gt_1'])  # or bbox_gt_2, they are the same

    if bbox_pred1 is None or bbox_pred2 is None or bbox_gt is None:
        continue  # skip rows with invalid bbox

    correct_1 = is_correct(bbox_pred1, bbox_gt)
    correct_2 = is_correct(bbox_pred2, bbox_gt)
    results.append((correct_1, correct_2))

# Build contingency table for McNemar test
table = np.zeros((2,2), dtype=int)
for c1, c2 in results:
    table[int(c1)][int(c2)] += 1

print("Contingency Table:")
print(table)

# McNemar Test
result = mcnemar(table, exact=True)
print("McNemar test statistic:", result.statistic)
print("p-value:", result.pvalue)
