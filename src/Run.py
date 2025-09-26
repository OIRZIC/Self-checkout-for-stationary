import os
import datetime
import cv2
import numpy as np
import onnxruntime as ort  # Thêm import này để sử dụng 'ort' thay cho 'onnxruntime'

from byte_tracker.tracker.byte_tracker import BYTETracker
import time
import database


# Danh sach cac lop cua mo hinh
class_names = [
        "BUT_CHI_DIXON",
        "BUT_HIGHLIGHT_MNG_TIM",
        "BUT_HIGHLIGHT_RETRO_COLOR",
        "BUT_LONG_SHARPIE_XANH",
        "BUT_NUOC_CS_8623",
        "BUT_XOA_NUOC",
        "HO_DOUBLE_8GM",
        "KEP_BUOM_19MM",
        "KEP_BUOM_25MM",
        "NGOI_CHI_MNG_0.5_100PCS",
        "SO_TAY_A6",
        "THUOC_CAMPUS_15CM",
        "THUOC_DO_DO",
        "THUOC_PARABOL",
        "XOA_KEO_CAPYBARA_9566"
    ]

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

class BYTETrackerArgs:
    def __init__(self, track_thresh, track_buffer, mot20, match_thresh,
                aspect_ratio_thresh, min_box_area):
        self.track_thresh        = track_thresh
        self.track_buffer        = track_buffer
        self.mot20               = mot20
        self.match_thresh        = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area        = min_box_area

# Các hàm liên quan đến phát hiện và vẽ kết quả
def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []

    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)
    keep_boxes = []

    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]
        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    return intersection_area / union_area

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def draw_box(image, box, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def draw_text(image, text, box, color=(0, 0, 255), font_size=0.001, text_thickness=2):
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)
    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)
    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

def draw_masks(image, boxes, classes, mask_alpha=0.3):
    mask_img = image.copy()
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()
    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)
    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]
        draw_box(det_img, box, color)
        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img
# Lớp YOLOv8
class YOLO_Model:
    def __init__(self, path, conf_thres=0.8, iou_thres=0.6):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.initialize_model(path)


    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = ort.InferenceSession(path, providers=ort.get_available_providers()) #lấy thông tin mô hình và lựa chọn nền tảng để chạy (CUDA Hoặc CPU)
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes


    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

def calculate_and_display_fps(frame, start_time, font_scale=0.5, font_thickness=1, color=(0, 255, 0)):
    """
    Calculate FPS and display it on the frame.

    Parameters:
        frame (numpy.ndarray): The video frame to overlay the FPS on.
        start_time (float): The time when the frame processing started.
        font_scale (float): Scale of the font for the FPS text.
        font_thickness (int): Thickness of the font.
        color (tuple): Color of the FPS text in BGR format.

    Returns:
        numpy.ndarray: Frame with FPS overlay.
    """
    # Calculate FPS
    end_time = time.time()
    time_diff = end_time-start_time
    if time_diff > 0:
        fps = 1 / (end_time - start_time)
        # Format FPS text
        fps_text = f"FPS: {fps:.2f}"

        # Get frame dimensions
        height, width, _ = frame.shape

        # Calculate position for top-right corner
        position = (width - 10 - len(fps_text) * 20, 30)  # Adjust position to be on the right

        # Overlay the FPS text on the frame
        cv2.putText(frame, fps_text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
        return frame, fps
    else:
        return None



def main():
    # model_path = r"H:\FINAL_ PROJECT\DATASET FOLDER\STATIONARY DATA\self-made-v7\best_training_output_v5s.onnx"
    # model_path = r"H:\FINAL_ PROJECT\DATASET FOLDER\STATIONARY DATA\self-made-v7\best_training_output_v8s.onnx"

    model_path = "../weights/yolov8n.onnx"
    yolo_detector= YOLO_Model(model_path,conf_thres=0.8,iou_thres=0.6)
    webcam =0
    cap = cv2.VideoCapture(webcam)

    product_prices= database.get_product_prices()

    #Create attribute of transaction storing place:
    transaction_id = 0
    os.makedirs("../receipts", exist_ok=True)
    print(f"Starting transaction ID: {transaction_id}")

    #Create attribute of FPS plot storing place:
    save_plot_folder = "../graphs"
    os.makedirs(save_plot_folder, exist_ok=True)


    # Prepare lists to store detected bounding box coordinates and scores
    detections = []
    detections_class_id = []

    track_id_to_class_id = {}
    track_id_count = {}  # Sử dụng dictionary để theo dõi số lần mỗi track_id xuất hiện

    class_name = ""


    #Khoi tao ByteRacker
    args = BYTETrackerArgs(track_thresh= 0.7,
                          track_buffer= 30,
                          mot20=True,
                          match_thresh=0.8,
                          aspect_ratio_thresh=0.5,
                          min_box_area= 20)
    obj_tracker = BYTETracker(args)
    transaction_id = 0
    os.makedirs("../receipts", exist_ok=True)
    print(f"Starting transaction ID: {transaction_id}")


    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        start_time = time.time()
        # initiate transaction
        transaction_file_path = os.path.join("../receipts", f"transaction_{transaction_id}.txt")

        boxes, scores, class_ids = yolo_detector(frame)

        frame               = cv2.resize(frame,(640,480))

        orig_frame          =frame.copy()

        frame_h, frame_w    = frame.shape[:2]

        frame_size          =np.array([frame_h,frame_w]) #extract frame size for BYTETRACK

        class_id_count = {}

        updated_text = []

        boxes, scores, class_ids = yolo_detector(frame)

        # Final shopping cart:
        cart = {}

        for box,score,class_id in zip(boxes,scores,class_ids):
            x1,y1,x2,y2 = box.astype(int)

            class_id = class_ids
            score = round(score,2)
            detected_class_id= class_id
            # detected_class_label = class_names[class_id] if 0 <= class_id < len(class_names) else "Unknown"

            detections.append([x1,y1,x2,y2, score])
            detections_class_id.append([detected_class_id,score])

            # print(f"Detected Object - Class ID: {detected_class_id}, Score : {score}, Box: ({x1}, {y1}, {x2}, {y2})")

            # #Draw bbox
            # cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # #Labels
            # label = f"{detected_class_label}: {score}"
            # cv2.putText(orig_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

            # Draw the detection results on the frame

        if len(detections)>0 and len(detections_class_id) >0:
            # Chuyển đổi detections thành numpy array, giả sử mỗi detection là (x, y, width, height, class_id)
            class_ids = [c[0] for c in detections_class_id]  # Lấy class_id từ các detections

            # Update_trackID
            track_ids = obj_tracker.update(np.array(detections), frame_size, frame_size)
            # Filter object detections based on tracking results


            print(track_ids)

            if track_ids is not None and len(track_ids) > 0:
                for i, track in enumerate(track_ids):
                    x1b, y1b, x2b,y2b  = track.tlbr
                    track_id = track.track_id

                    # Kiểm tra nếu kích thước bounding box hợp lệ

                    # Cập nhật số lần track_id xuất hiện
                    if track_id not in track_id_count:
                        track_id_count[track_id] = 0
                    track_id_count[track_id] += 1

                    # Nếu track_id đã được đếm đủ (từ 10 đến 15 lần), vẽ bounding box
                    if 0<= track_id_count[track_id] <= 10:
                        # Liên kết track_id với class_id

                        # class_id = int(class_ids[i])
                        if isinstance(class_ids[i], np.ndarray):
                            class_id = int(class_ids[i].item())
                        else:
                            class_id = int(class_ids[i])

                        if class_id in class_id_count:
                            class_id_count[class_id] += 1  # Tăng số lượng nếu đã tồn tại
                        else:
                            class_id_count[class_id] = 1  # Khởi tạo số lượng nếu chưa tồn tại

                        # # In ra thông tin track_id
                        # print(f"Track ID: {track_id}, Class ID: {class_id}, Class Name: {class_name}\nCoordinates: ({x1b:.2f}, {y1b:.2f}, {x2b:.2f}, {y2b:.2f})")
                        # for class_id, count in class_id_count.items():
                        #     print(f"Product Count: {class_name} x {int(round(count/10,1))} ")


                        #verify count
                        count = int(round(class_id_count[class_id] / 10, 1))
                        if count>0:
                            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else "Unknown"
                            count = count #Product Price
                            price = round(product_prices[class_name])

                            if class_id not in cart:
                                # Nếu class_id chưa tồn tại, thêm mới
                                cart[class_id] = {"class_name": class_name, "count": count, "price": price}
                            else:
                                # Nếu class_id đã tồn tại, chỉ cập nhật số lượng
                                cart[class_id]["count"] += count

                            updated_text.append(f"{class_name}: {count} x {price} VND")
                            print(f"{class_name}: {count} x {price} VND\n")
                        # bounding box drawing for tracked_objects
                        cv2.rectangle(orig_frame,(int(x1b), int(y1b)),(int(x2b), int(y2b)),
                                      (0, 255, 0),1
                        )
                        cv2.putText(orig_frame,f"Prod-Code: {class_id}, Prod-name {class_name}, Score {scores}",(int(x1b), int(y1b) - 10),cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,(0, 255, 0),1
                        )


                y_offset = 10  # Bắt đầu từ 30px
                for text in updated_text:
                    y_offset += 25   # Cách nhau 30px cho mỗi dòng
                    cv2.putText(orig_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print('No product detected')
            # se them doan tu xoa track_id neu nhu vat the do khong con xuat hien

        orig_frame, fps = calculate_and_display_fps(orig_frame,start_time)

        cv2.imshow("Detected_object",orig_frame)

        # Press Q to exit program:
        total_price = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # #Get Date and Time

            now_time = datetime.datetime.now()
            # Transaction generator code:
            with open(transaction_file_path, "w", encoding="utf-8") as file:
                file.write(f"STATIONARY SHOP\nBill exported at {now_time}\n")
                file.write(f"Transaction ID: {transaction_id}\n")
                file.write("--------------------------------------------------\n")
                for class_id, product_info in cart.items():
                    class_name = product_info["class_name"]
                    count = product_info["count"]
                    price = product_info["price"]
                    total_price += count*price

                # Ghi vào file
                file.write(f"{class_name}  :\t\t{count} \t{price}\n\n")

                # Ghi tổng giá trị đơn hàng
                file.write(f"Total price:\t\t\t\t\t\t{total_price}\n")
                file.write(f"Bill saved :\t\t\t\t{transaction_file_path}\n")
                print(f"Total Price: {total_price}")

            new_session = input("Start new session? (Y/N): ").strip().upper()

            if new_session != "Y":
                print("Closing program...")
                break  # Thoát khỏi vòng lặp
if __name__ == "__main__":
    main()


