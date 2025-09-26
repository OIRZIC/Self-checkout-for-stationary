from ultralytics import YOLO
import numpy as np
import cv2
import database
import datetime
import os

import matplotlib.pyplot as plt
import time


import torch
from byte_tracker.tracker.byte_tracker import BYTETracker

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# load yolo model and make it run on GPU instead of CPU
model = YOLO("../weights/best.pt").to(device)

class BYTETrackerArgs:
    def __init__(self, track_thresh, track_buffer, mot20, match_thresh,
                aspect_ratio_thresh, min_box_area):
        self.track_thresh        = track_thresh
        self.track_buffer        = track_buffer
        self.mot20               = mot20
        self.match_thresh        = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area        = min_box_area


def calculate_and_display_fps(frame, start_time, font_scale=0.7, font_thickness=2, color=(0, 255, 0)):
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


def export_transaction_bill(transaction_file_path,transaction_id,class_counts,product_prices):
    """

    :param transaction_file_path:       path to save file
    :param transaction_id:              unique id for transaction
    :param class_counts:                 Dictionary of product and their count
    :param product_price:               Dictionary of product and their price


    :return:float:                      Total price of transaction
    """

    total_price = 0
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Open the file to write
    with open(transaction_file_path, "w", encoding="utf-8") as file:
        # Write the bill header
        file.write(f"STATIONARY SHOP\nBill exported at {current_time}\n")
        file.write(f"Transaction ID: {transaction_id}\n")
        file.write("--------------------------------------------------\n")

        # Write product details and calculate total price
        for product, count in class_counts.items():
            if count > 0:
                price = product_prices.get(product, 0)
                total_price += count * price
                file.write(f"{product}  :\t\t{count} \t{price}\n\n")

        # Write total price and summary
        file.write(f"Total price:\t\t\t\t\t\t{total_price}\n")
        file.write(f"Bill saved at:\t\t\t\t{os.path.abspath(transaction_file_path)}\n")

    print(f"Transaction bill saved successfully at {transaction_file_path}")
    return total_price


def count_object(video_path):
    assert type(video_path)       == int, "video_path argument should be string"
    product_prices=    database.get_product_prices()

    args = BYTETrackerArgs(track_thresh= 0.7,
                          track_buffer= 30,
                          mot20=True,
                          match_thresh=0.8,
                          aspect_ratio_thresh=0.8,
                          min_box_area= 10)
    #Create attribute of transaction storing place:
    transaction_id = 0
    os.makedirs("../receipts", exist_ok=True)
    print(f"Starting transaction ID: {transaction_id}")

    #Create attribute of FPS plot storing place:
    save_plot_folder = "../graphs"
    os.makedirs(save_plot_folder, exist_ok=True)

    obj_tracker = BYTETracker(args)
    cap = cv2.VideoCapture(video_path)

    #dictionary for products

    class_counts = {name: 0 for name in product_prices.keys()}
    tracked_ids = set()
    tracked_product_ids = set()  # Set to store IDs of counted products

    missing_frame_counts= {}

    # FPS
    fps_list = []  # Save FPS at frame

    class_name = ""

    while True:
        if not cap.isOpened():
            break
        else:
            start_time = time.time()

            #initiate transaction
            transaction_file_path = os.path.join("../receipts", f"transaction_{transaction_id}.txt")

            ret, frame  = cap.read()
            results      = model.predict(frame)
            first_result = results[0]
            labels       = first_result.names


            # Dictionaries define
            # Prepare lists to store detected bounding box coordinates and scores
            detections = []
            track_ids = []
            confidences=[]

            fps_list = []
            # List of detected products
            detected_products = []  # List to store names of detected products
            if ret:
                transaction_id += 1
                frame               = cv2.resize(frame, (640,480))
                orig_frame          = frame.copy() # we take a copy of our original frame before loosing it.
                #Get height, width of frame
                frame_h, frame_w    = frame.shape[:2]
                frame_size          = np.array([frame_h, frame_w])

                for result in results:
                    for box,r in zip(result.boxes, result.boxes.data):
                        x,y,w,h     = box.xywh[0]
                        class_id    = int(box.cls[0])
                        class_name  = labels[class_id] if labels[class_id] else ""

                        #Get products name - price
                        if class_name in product_prices:
                            price = product_prices[class_name]
                            print(f"Product: {class_name}, Price: {price}")

                        x1,y1,x2,y2 = int(x - w / 2), int(y - h / 2),int(x + w / 2),int(y + h / 2)

                        score = round(float(r[-2].item()), 2)  # Round to 2 decimal places
                        track_id= int(r[-1].item())

                        detections.append([x1, y1, x2, y2, score]) # xyxy and score

                        track_ids.append(track_id)
                        confidences.append(score)

                        # print(f"Detection: [{x1}, {y1}, {x2}, {y2},{score}]")
                        # print(f"Track id : {track_ids}")
                        # print(f"Confidence : {confidences}")

                #BYTE TRACK GENERATIONS
                if len(detections) > 0:
                    track_ids = obj_tracker.update(np.array(detections), frame_size, frame_size)

                    # Get tracking coordinate from YOLO     detections
                    for track in track_ids:
                        x1b, y1b, width_b, height_b = track.tlwh
                        x2b, y2b = x1b + width_b, y1b + height_b
                        track_id = track.track_id

                        if track_id not in tracked_ids:
                            tracked_ids.add(track_id)  # Processed Track

                            if track_id not in tracked_product_ids:  # Ensure we don't count the same product again
                                class_counts[class_name] += 1
                                tracked_product_ids.add(track_id)  # Mark this product as counted

                        # Draw bbox of Byte-tracking frame
                        cv2.rectangle(orig_frame, (int(x1b), int(y1b)), (int(x2b), int(y2b)), (0, 255, 0), 5)

                        # ad
                        detected_products.append(class_name)
                        #  List of products in frame:
                        y_offset = 50
                        for product, count in class_counts.items():
                            if count > 0:
                                cv2.putText(orig_frame, f"{product}: {count}", (50, y_offset),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                                            thickness=2)
                                y_offset += 30
                        total_price = 0
                        for product, count in class_counts.items():
                            if count > 0:
                                price = product_prices.get(product, 0)
                                total_price += count * price
                                print(f"total price :{total_price}")

                        # Check to remove item
                        for track_id in list(tracked_ids):
                            if track_id not in [track.track_id for track in track_ids]:  # If track is not exist in new frame
                                # Reset all the counter till detected objects no longer exist in new frame
                                if track_id in missing_frame_counts:
                                    missing_frame_counts[track_id] +=1
                                else:
                                    missing_frame_counts[track_id] =0

                                if missing_frame_counts[track_id] >10:
                                    if track_id in class_counts:
                                        class_counts[track_id]=0
                                    tracked_ids.remove(track_id)
                                    del missing_frame_counts[track_id]
                            else:
                                if track_id in missing_frame_counts:
                                    del missing_frame_counts[track_id]


                                # Code cũ
                                # if track_id in class_counts:
                                #     class_counts[class_name] = 0
                                # tracked_ids.remove(track_id)
                orig_frame, fps = calculate_and_display_fps(orig_frame,start_time)
                cv2.imshow("Frame",orig_frame)
                # press esc for quitting the video

                total_price = 0

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    #Get date and time
                    current_time = datetime.datetime.now()
                    # Bill generator code:
                    with open(transaction_file_path,"w",encoding="utf-8") as file:
                        file.write(f"STATIONARY SHOP\nBill exported at {current_time}\n")
                        file.write(f"Transaction ID: {transaction_id}\n")
                        file.write("--------------------------------------------------\n")
                        for product, count in class_counts.items():
                            if count > 0:
                                price = product_prices.get(product, 0)
                                total_price += count * price

                                file.write(f"{product}  :\t\t{count} \t{price}\n\n")
                                file.write(f"total price:\t\t\t\t\t\t{total_price}\n")
                                file.write(f"Bill saved :\t\t\t\t{transaction_file_path}\n")
                                print("\n\n\n")
                                break
                    # total = export_transaction_bill(transaction_file_path, transaction_id, class_counts, product_prices)
                    cap.release()
                    cv2.destroyAllWindows()

                    plot_file_path = os.path.join(save_plot_folder, "fps_plot.png")  # File path to save the plot
                    plt.savefig(plot_file_path)  # Save the plot as a PNG file in the specified folder
                    plt.close()  # Close the plot to free up memory

                    new_session = input("Do you want to start a new session? (y/n): ").strip().lower()
                    if new_session !="y":
                        print("Exiting program")
                        break
                    else:
                        print("Staring new seasons")
def main():
    count_object(0)

if __name__ == "__main__":
    main()

# if len(detections) > 0:
#     track_ids = obj_tracker.update(np.array(detections),frame_size,frame_size)
#     for track in track_ids:
#         # Extract bounding box in (top-left x, top-left y, width, height) format
#         x1b, y1b, width_b, height_b = track.tlwh
#         x2b, y2b = x1b + width_b, y1b + height_b  # Bottom-right corne
#         track_id = track.track_id
#
#         if track_id not in tracked_ids:
#             tracked_ids.add(track_id)  # Mark this track_id as processed
#             if track_id is not None:
#                 class_counts[class_name] += 1  # Increment the count for the detected class
#         # Draw bounding box on the Byte-tracking-frame
#         cv2.rectangle(orig_frame, (int(x1b), int(y1b)), (int(x2b), int(y2b)), (0, 255, 0),
#                       2)
#
#         detected_products.append(class_name)  # Add detected product name to list
