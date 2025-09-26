from ultralytics import YOLO

model_path = r"D:\Final\SELF-CHECKOUT-PROJECT\SELFCHECKOUTSYSTEM\weights\yolov8n.pt"
model = YOLO(model_path, task="detect")
out_file = model.export(
    format="onnx",
    imgsz=640,
    device = [0],
    int8 = False
)
# reference https://docs.ultralytics.com/modes/export
