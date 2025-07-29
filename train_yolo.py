from ultralytics import YOLO
import torch

if __name__ == "__main__":

    model = YOLO("yolov8n.pt")  # 加载预训练的 yolov8n 检测模型

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # train
    train_results = model.train(
        data="./yolopng/ZZZUI2.v1i.yolov8-obb/data.yaml",  # 配置文件
        epochs=512,
        batch=64,
        imgsz=640,
        device=device,
        workers=0,  # win 上设为 0 避免多进程问题
        name="whole_image_detection4",
    )

    """
    EarlyStopping: Training stopped early as no improvement observed in last 100 epochs. Best results observed at epoch 9, best model saved as best.pt.
    To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
    绷不住了

    v3:
    EarlyStopping: Training stopped early as no improvement observed in last 100 epochs. Best results observed at epoch 4, best model saved as best.pt.
    To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

    更难绷

    UI2v1
    EarlyStopping: Training stopped early as no improvement observed in last 100 epochs. Best results observed at epoch 292, best model saved as best.pt.
    To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.   

    392 epochs completed in 1.564 hours.
    Optimizer stripped from runs\detect\whole_image_detection4\weights\last.pt, 6.3MB
    Optimizer stripped from runs\detect\whole_image_detection4\weights\best.pt, 6.3MB
    这还行
    """

    # 评估模型
    metrics = model.val()

    # 重新导出 onnx
    # model = YOLO("runs/detect/whole_image_detection/weights/best.pt")

    path = model.export(format="onnx", opset=15, dynamic=True)
    print(f"模型已导出至: {path}")
