import cv2
import os
import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop',  'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

FILTER = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

SCORE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.25
MAX_OUTPUT_SIZE_PER_CLASS = 100
MAX_TOTAL_SIZE = 100
MODEL_PATH = os.path.normpath("../models/yolov4.tflite")


class TFLite():
    def __init__(self, width, height, model_path=MODEL_PATH):
        self.width = width
        self.height = height

        self.interpreter = tf.lite.Interpreter(model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()

    def predict(self, img, vis=True):
        """Predicts bounding boxes, scores and classes from a given image.

        Args:
            img (np.array): the image to be predicted.
            vis (bool, optional): Whether we want to return a visualization image. Defaults to True.

        Returns:
            (boxes, scores, classes, detections, vis_image): the prediction and the visualized image.
        """
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        inp = tf.image.resize(inp, (self.height, self.width))
        inp = tf.expand_dims(inp, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()
        pred = [self.interpreter.get_tensor(self.output_details[i]['index'])
                for i in range(len(self.output_details))]
        boxes, pred_conf = filter_boxes(
            pred[0], pred[1],
            score_threshold=SCORE_THRESHOLD,
            input_shape=tf.constant([self.height, self.width]),
        )
        boxes, scores, classes, detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf,
                              (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=MAX_OUTPUT_SIZE_PER_CLASS,
            max_total_size=MAX_TOTAL_SIZE,
            iou_threshold=IOU_THRESHOLD,
            score_threshold=SCORE_THRESHOLD,
        )

        pred_bbox = [boxes.numpy(), scores.numpy(),
                     classes.numpy(), detections.numpy()]
        filtered_bbox = process_bbox(pred_bbox)

        if vis:
            vis_image = vis_bbox(img, filtered_bbox)
        else:
            vis_image = None

        return filtered_bbox, vis_image


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape=tf.constant([416, 416])):
    """Filters the bounding boxes according to the scores.

    Args:
        box_xywh (_type_): _description_
        scores (_type_): _description_
        score_threshold (float, optional): _description_. Defaults to 0.4.
        input_shape (tf.constant, optional): input shape of the model. Defaults to tf.constant([416, 416]).

    Returns:
        (_type_): _description_
    """
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(
        class_boxes,
        [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(
        pred_conf,
        [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    return (boxes, pred_conf)


def process_bbox(bboxes):
    filtered = []
    out_boxes, out_scores, out_classes, num_boxes = bboxes
    count = 0
    for i in range(num_boxes[0]):
        class_idx = int(out_classes[0][i])
        if class_idx < 0 or class_idx > len(CLASSES) or CLASSES[class_idx] not in FILTER:
            continue
        count += 1
        filtered.append(
            (out_boxes[0][i], out_scores[0][i], out_classes[0][i], i))
    return filtered


def vis_bbox(image, bboxes):
    image_h, image_w, _ = image.shape
    # out_boxes, out_scores, out_classes, num_boxes = bboxes

    for coor, out_score, out_class, i in bboxes:
        class_idx = int(out_class)

        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
        cv2.rectangle(image, c1, c2, (255, 0, 0), bbox_thick)

        bbox_mess = '%s: %.2f' % (CLASSES[class_idx], out_score)
        t_size = cv2.getTextSize(
            bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(image, c1, (int(c3[0]), int(
            c3[1])), (255, 0, 0), -1)  # filled

        cv2.putText(image, bbox_mess,
                    (c1[0], int(c1[1] - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    
    return image
