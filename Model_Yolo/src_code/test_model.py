from ultralytics import YOLO
import cv2
import math

# label book clock curtain painting vase tv
# train 1869 1435  1925    2401     1249 4337 = 13216
# val   551  348   201     160      261  403 = 1924

# # Model
# model = YOLO("Project_1/runs/detect/train/weights/best.pt")  # Tạo mô hình YOLO với trọng số tốt nhất
# results = model("D:/PyCharm/Python/Computer_Vision/Advance/Project_1/test/test_img", save=True, conf=0.5)

# ==============================Camera==============================
# # Object classes
# # Start webcam
# cap = cv2.VideoCapture(0)  # Khởi tạo kết nối với webcam
# model = YOLO("Project_1/runs/detect/train/weights/best.pt")  # Tạo mô hình YOLO với trọng số tốt nhất
# classNames = ['book','clock','curtain','painting','vase','tv']  # Danh sách các lớp đối tượng, ở đây chỉ có 'book'
#
# while True:
#     success, img = cap.read()  # Đọc frame từ webcam
#     results = model(img, stream=True)  # Dự đoán đối tượng trong frame
#
#     # Process results
#     max_confidence = 0.5  # Độ tin cậy tối thiểu để vẽ bounding box
#     best_box = None  # Biến lưu trữ box có độ tin cậy cao nhất
#
#     for r in results:
#         boxes = r.boxes
#
#         for box in boxes:
#             # Confidence
#             confidence = box.conf[0]
#             print("Confidence --->", confidence)
#
#             # Check if the class is "book" and confidence is higher than the threshold
#             cls = int(box.cls[0])
#             if classNames[cls] in ['book','clock','curtain','painting','vase','tv'] and confidence > max_confidence:
#                 max_confidence = confidence
#                 best_box = box  # Cập nhật box có độ tin cậy cao nhất
#
#     # Draw bounding box for the best box (if any)
#     if best_box is not None:
#         x1, y1, x2, y2 = best_box.xyxy[0]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#         # Draw bounding box
#         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#
#         # Class name
#         cls = int(best_box.cls[0])
#         print("Class name -->", classNames[cls])
#
#         # Object details
#         org = [x1, y1]
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         fontScale = 1
#         color = (255, 0, 0)
#         thickness = 2
#
#         cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
#
#     cv2.imshow('Webcam', img)
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ hiển thị của OpenCV


# ==============================Video==============================
# Model
model = YOLO("runs/detect/train/weights/best.pt")  # Tạo mô hình YOLO với trọng số tốt nhất
results = model("D:/PyCharm/Python/Computer_Vision/Advance/Project_1/test/test_video", save=True, conf=0.5, vid_stride=3, save_frames=True, save_crop=True)
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs
