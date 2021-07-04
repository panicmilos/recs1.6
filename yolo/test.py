import numpy as np
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt


TEST_FILE_PATH = "training/test_data.csv"


def calc_iou(gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox

    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p,
                             y_bottomright_gt)

    # if the GT bbox and predcited BBox do not overlap then iou=0
    if (x_bottomright_gt < x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox

        return 0.0
    if (
            y_bottomright_gt < y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox

        return 0.0
    if (
            x_topleft_gt > x_bottomright_p):  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox

        return 0.0
    if (
            y_topleft_gt > y_bottomright_p):  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox

        return 0.0

    GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
    Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)

    x_top_left = np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

    intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)

    return intersection_area / union_area


def calc_precision_recall(image_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    precision = 0
    recall = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0
    for img_id, res in image_results.items():
        true_positive += res['true_positive']
        false_positive += res['false_positive']
        false_negative += res['false_negative']
        try:
            precision = true_positive/(true_positive+ false_positive)
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = true_positive/(true_positive + false_negative)
        except ZeroDivisionError:
            recall = 0.0
    return precision, recall


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou(gt_box, pred_box)

            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort) == 0:
        tp = 0
        fp = 0
        fn = 0
        return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in iou_sort:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}


def parse_test_data():
    with open(TEST_FILE_PATH, 'r') as input_file_stream:
        lines = input_file_stream.readlines()

    parsed_test_data = []
    for line in lines:
        test_data_map = {}
        line = line.strip().split(" ")

        test_data_map["image"] =  line[0]
        test_data_map["boxes"] = []
        test_data_map["classes"] = []
        for box in line[1:]:
            box = box.split(",")
            test_data_map["boxes"].append(int(coordinate) for coordinate in box[:4])
            test_data_map["classes"].append(int(box[4]))

        parsed_test_data.append(test_data_map)

    return parsed_test_data


if __name__ == "__main__":
    parsed_test_data = parse_test_data()
    [print(data) for data in parsed_test_data]
