import numpy as np
from yolo import YOLO
from Utils.utils import detect_object


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


def calc_precision_recall(img_results):
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
    for img_id, res in img_results.items():
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


def get_single_image_results(gt_boxes, pred_boxes, iou_thr=0.5):
    if len(gt_boxes) == 0:
        return {'true_positive': 0, 'false_positive': len(pred_boxes), 'false_negative': 0}
    if len(pred_boxes) == 0:
        return {'true_positive': 0, 'false_positive': 0, 'false_negative': len(gt_boxes)}

    output = {'true_positive': 0, 'false_positive': 0, 'false_negative': 0}
    matched_gt_boxes = {gt_box: False for gt_box in gt_boxes}

    for pred_box in pred_boxes:
        matched = False
        for gt_box in gt_boxes:
            if matched_gt_boxes[gt_box]:
                continue
            iou = calc_iou(gt_box, pred_box)

            if iou > iou_thr:
                output['true_positive'] += 1
                matched = True
                matched_gt_boxes[gt_box] = True
                break
        if not matched:
            output['false_positive'] += 1

    output['false_negative'] = len(gt_boxes) - output['true_positive']
    return output


def calculate_tp_fp_fn_for_single_class_and_image(image, expected_boxes, expected_classes, test_class, yolo):
    prediction_boxes, _ = detect_object(yolo, image, save_img=False, save_img_path="", postfix="")
    adapted_predicted_boxes = [box[0:4] for box in prediction_boxes if box[4] == test_class]
    adapted_expected_boxes = [box for i, box in enumerate(expected_boxes) if expected_classes[i] == test_class]

    return get_single_image_results(adapted_expected_boxes, adapted_predicted_boxes)


def calculate_precision_recall_accuracy(data):
    yolo = YOLO(
        **{
            "model_path": "model_data/yolo-final.h5",
            "anchors_path": "model_data/yolo_anchors.txt",
            "classes_path": "model_data/data_classes.txt",
            "score": 0.25,
            "gpu_num": 1,
            "model_image_size": (416, 416),
        }
    )

    results = {"0": {}, "1": {}}
    total_correct_predictions = 0
    total_attempted_predictions = 0

    for test_class in [0, 1]:
        img_results = {}
        for test in data:
            img_results[test["image"]] = calculate_tp_fp_fn_for_single_class_and_image(test["image"], test["boxes"],
                                                                                       test["classes"], test_class, yolo)
            total_correct_predictions += img_results[test["image"]]["true_positive"]
            total_attempted_predictions += img_results[test["image"]]["true_positive"]
            total_attempted_predictions += img_results[test["image"]]["false_positive"]
            total_attempted_predictions += img_results[test["image"]]["false_negative"]

        results[str(test_class)]["precision"], results[str(test_class)]["recall"] = calc_precision_recall(img_results)

    accuracy = 100 * total_correct_predictions / total_attempted_predictions
    return results, accuracy


def parse_test_data():
    with open(TEST_FILE_PATH, 'r') as input_file_stream:
        lines = input_file_stream.readlines()

    parsed_test_data = []
    for line in lines:
        test_data_map = {}
        line = line.strip().split(" ")

        test_data_map["image"] = line[0]
        test_data_map["boxes"] = []
        test_data_map["classes"] = []
        for box in line[1:]:
            box = box.split(",")
            test_data_map["boxes"].append([int(coordinate) for coordinate in box[:4]])
            test_data_map["classes"].append(int(box[4]))

        parsed_test_data.append(test_data_map)

    return parsed_test_data


def print_results(found_results, accuracy):
    print("*******************************************************")
    print("Total accuracy : " + accuracy + "%")
    print("----------------------------")
    print("Counter precision : " + str(found_results["0"]["precision"] * 100) + "%")
    print("Counter recall : " + str(found_results["0"]["recall"] * 100) + "%")
    print("----------------------------")
    print("Terror precision : " + str(found_results["1"]["precision"] * 100) + "%")
    print("Terror recall : " + str(found_results["1"]["recall"] * 100) + "%")
    print("*******************************************************")


if __name__ == "__main__":
    test_data = parse_test_data()
    results, calculated_accuracy = calculate_precision_recall_accuracy(test_data)
    print_results(results, accuracy)
