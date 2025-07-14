from ultralytics import YOLO
import torch

if __name__ == "__main__":

    model = YOLO("yolov8n.pt")  # 加载预训练的 yolov8n 检测模型

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # train
    train_results = model.train(
        data="./yolopng/ZZZUI.v2i.yolov8-obb/data.yaml",  # 配置文件
        epochs=512,
        batch=64,
        imgsz=640,
        device=device,
        workers=0,  # win 上设为 0 避免多进程问题
        name="whole_image_detection",
    )

    """
    EarlyStopping: Training stopped early as no improvement observed in last 100 epochs. Best results observed at epoch 9, best model saved as best.pt.
    To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
    绷不住了
    """

    # 评估模型
    metrics = model.val()

    # 重新导出 onnx
    # model = YOLO("runs/detect/whole_image_detection/weights/best.pt")

    path = model.export(format="onnx", opset=15, dynamic=True)
    print(f"模型已导出至: {path}")
