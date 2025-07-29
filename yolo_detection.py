# yolo_detection.py

import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import torch
import config

from torchvision.ops import nms

model_path = "runs/detect/whole_image_detection4/weights/best.onnx"
session = ort.InferenceSession(
    model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]
class_map_file = "./yolopng/ZZZUI2.v1i.yolov8-obb/class_map.txt"
try:
    with open(class_map_file, "r", encoding="utf-8") as f:
        MY_CLASS_NAMES = [line.strip() for line in f.readlines()]
    print(f"[Info/yolo] 成功从 {class_map_file} 加载了 {len(MY_CLASS_NAMES)} 个类别。")
    print("[Info/yolo] 类别列表:", MY_CLASS_NAMES)
except FileNotFoundError:
    print(f"[Error/yolo] 类别文件未找到: {class_map_file}")
    sys.exit(1)

"""
class_name

chainatk # 连携技
clock # 右上角闹钟
qable # 能开大
qdisable # 不能开大
settings # ESC 界面的设置
"""


def preprocess_batch(images, imgsz=640):
    """
    对一个 batch 的图像进行打包预处理，image 为 [height, weight, bgr]
    """
    batch_tensors = []
    for image in images:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        scale = imgsz / max(img_height, img_width)
        new_w, new_h = int(img_width * scale), int(img_height * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        input_tensor = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        pad_h = (imgsz - new_h) // 2
        pad_w = (imgsz - new_w) // 2
        input_tensor[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_img

        input_tensor = input_tensor.transpose(2, 0, 1)  # [h,w,c] -> [c,h,w]
        input_tensor = input_tensor.astype(np.float32) / 255.0
        batch_tensors.append(input_tensor)

    return np.stack(batch_tensors, axis=0)  # 将单张图像张量堆叠成一个批次


def postprocess_batch(detections, conf_threshold):
    """
    对 batch 的输出进行后处理
    """
    results = []

    detections = detections.transpose(0, 2, 1)

    # [batch_size, num_classes + 4, num_proposals] -> [batch_size, num_proposals, num_classes + 4]
    for single_image_detection in detections:
        scores_all_classes = single_image_detection[..., 4:]
        scores = np.max(scores_all_classes, axis=-1)
        class_ids = np.argmax(scores_all_classes, axis=-1)

        mask = scores > conf_threshold
        filtered_scores = scores[mask]
        filtered_class_ids = class_ids[mask]

        if len(filtered_class_ids) > 0:
            idx = np.argmax(filtered_scores)
            result_score = float(filtered_scores[idx])
            result_class_name = MY_CLASS_NAMES[filtered_class_ids[idx]]
            results.append((result_class_name, result_score))
        else:
            results.append(("NaC", -1.0))  # Not a Class

    return results


def detect_batch(images_to_detect, imgsz=640, conf_threshold=config.conf_threshold):
    """
    对 batch 进行 yolo 检测。
    """
    if not images_to_detect:
        return []

    input_batch = preprocess_batch(images_to_detect, imgsz)

    detections_batch = session.run(output_names, {input_name: input_batch})[0]

    results = postprocess_batch(detections_batch, conf_threshold)

    return results


def detect_image(image, imgsz=640):
    """
    检测图是啥，单张。
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_height, img_width = img.shape[:2]

    # 计算缩放比例，使最长边为 imgsz
    scale = imgsz / max(img_height, img_width)
    new_w, new_h = int(img_width * scale), int(img_height * scale)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建一个灰色的底板 ( yolo 常用的 letterbox 颜色)
    input_tensor = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)

    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2

    # 将缩放后的图片粘贴到底板中心
    input_tensor[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_img

    # resized_image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    # 转换为模型需要的格式: BGR->RGB, HWC->CHW, Normalization
    # input_tensor = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
    input_tensor = input_tensor.astype(np.float32) / 255.0

    detections = session.run(output_names, {input_name: input_tensor})[0]
    detections = detections.transpose(0, 2, 1)[0]  # [1, 5, 8400] -> [8400, 5]
    scores_all_classes = detections[..., 4:]
    scores = np.max(scores_all_classes, axis=-1)
    class_ids = np.argmax(scores_all_classes, axis=-1)
    mask = scores > config.conf_threshold
    filtered_scores = scores[mask]
    filtered_class_ids = class_ids[mask]
    if len(filtered_class_ids):
        idx = np.argmax(filtered_scores)
        result_score = float(filtered_scores[idx])
        result_class_name = MY_CLASS_NAMES[filtered_class_ids[idx]]
    else:
        result_score = -1
        result_class_name = "NaC"
    return result_class_name, result_score


def test_image():
    image = cv2.imread("yolopng/test/3.jpg")
    print(detect_image(image))


# test_image()
