import numpy as np
import sys
import cv2
from ultralytics import YOLO
import traceback

# Thư viện PyQt5
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QFileInfo
from PyQt5.QtGui import QPixmap, QKeyEvent

import apriori_nt
# Các file cần thiết trong Project
from GUI_PyQt5.app_ui import Ui_MainWindow
from ChatBot.chatgui import chatbot_response

from apriori_nt import find_frequent_itemsets, extended_transactions, suggest_items


class CaptureVideo(QThread):
    signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal(list)

    def __init__(self, file_path, main_window):
        self.file_path = file_path
        self.cap = None
        self.model = YOLO("Model_Yolo/src_code/runs/detect/train/weights/best.pt")
        self.classNames = ['book', 'clock', 'curtain', 'painting', 'vase', 'tv']
        self.class_name_map = {'book': 'Sách', 'clock': 'Đồng hồ', 'curtain': 'Rèm', 'painting': 'Bức tranh', 'vase': 'Bình hoa', 'tv': 'TV'}
        self.main_window = main_window
        self.keep_running = True
        super().__init__()

    def run(self):
        detected_objects_history = []  # Danh sách lưu trữ các vật đã nhận dạng từ các khung hình trước đó
        try:
            if self.file_path is not None:
                self.cap = cv2.VideoCapture(self.file_path)
            else:
                self.cap = cv2.VideoCapture(0)
            while True and self.keep_running:
                ret, img = self.cap.read()
                if ret:
                    results = self.model(img, stream=True)
                    best_boxes = []
                    detected_objects = []
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            confidence = box.conf[0]
                            cls = int(box.cls[0])
                            if self.classNames[cls] in ['book', 'clock', 'curtain', 'painting', 'vase', 'tv']:
                                best_boxes.append(box)
                                detected_objects.append(
                                    self.classNames[cls])  # Thêm các vật nhận dạng được vào danh sách
                    detected_objects_history.extend(detected_objects)
                    for best_box in best_boxes:
                        x1, y1, x2, y2 = best_box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cls = int(best_box.cls[0])
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2
                        cv2.putText(img, self.classNames[cls], org, font, fontScale, color, thickness)
                    self.signal.emit(img)
                    self.finished_signal.emit(detected_objects_history)
                else:
                    break
        except Exception as e:
            print("Lỗi trong luồng chụp video:", e)
        finally:
            if self.keep_running and (self.file_path is not None or self.file_path is None):
                self.finished_signal.emit(detected_objects_history)

    def stop(self):
        self.keep_running = False
        self.terminate()

# Định nghĩa lớp MainWindow để quản lý giao diện chính
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.selected_image_file = ""
        self.detected_objects = []
        self.thread = None
        self.model = YOLO("Model_Yolo/src_code/runs/detect/train/weights/best.pt")
        self.classNames = ['book', 'clock', 'curtain', 'painting', 'vase', 'tv']
        self.class_name_map = {'book': 'Sách', 'clock': 'Đồng hồ', 'curtain': 'Rèm', 'painting': 'Bức tranh', 'vase': 'Bình hoa', 'tv': 'TV'}
        self.transactions = apriori_nt.transactions
        self.additional_items = apriori_nt.additional_items
        self.additional = apriori_nt.additional

        # Kết nối các sự kiện với các hàm tương ứng
        self.ui.ha_pushButton.clicked.connect(self.original_image)
        self.ui.ha_pushButton.clicked.connect(self.file_info)
        self.ui.video_pushButton.clicked.connect(self.start_capture_video)
        self.ui.video_pushButton.clicked.connect(self.file_info)
        self.ui.camera_pushButton.clicked.connect(self.start_capture_camera)
        self.ui.stop_pushButton.clicked.connect(self.stop_capture_video)
        self.ui.clear_pushButton.clicked.connect(self.clear_ui)
        self.ui.gy_nt_pushButton.clicked.connect(self.suggest_detected_objects)
        self.ui.delete_chat_pushButton.clicked.connect(self.delete_chat)

        # Kết nối sự kiện khi nhấn phím Enter trong enter_textEdit
        self.ui.enter_textEdit.installEventFilter(self)
        # Kết nối sự kiện khi nhấn nút enter_pushButton
        self.ui.enter_pushButton.clicked.connect(self.send_message)

    def stop_capture_video(self):
        try:
            if self.thread:
                self.thread.stop()
                self.thread.wait()
                self.thread = None
        except Exception as e:
            print("Lỗi khi dừng chụp video:", e)
        return self.thread

    # Bắt đầu chụp video từ tệp đã chọn
    def start_capture_video(self):
        if not self.stop_capture_video():
            file_path = self.video_or_camera()
            if file_path:
                self.thread = CaptureVideo(file_path, self)
                self.thread.start()
                self.thread.signal.connect(self.show_webcam)
                self.thread.finished_signal.connect(self.update_detected_objects_camera_video)

    # Bắt đầu chụp video từ camera
    def start_capture_camera(self):
        try:
            if not self.stop_capture_video():
                self.thread = CaptureVideo(0, self)
                self.thread.start()
                self.thread.signal.connect(self.show_webcam)
                self.thread.finished_signal.connect(self.update_detected_objects_camera_video)
        except Exception as e:
            print("Lỗi khi bắt đầu luồng chụp camera:", e)

    # Hiển thị hình ảnh từ webcam
    def show_webcam(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.ui.original_label.setPixmap(qt_img)

    # Chuyển đổi hình ảnh từ OpenCV sang QPixmap
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(990, 660, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    # Chọn tệp video từ máy tính hoặc camera
    def video_or_camera(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn tệp video", "", "Tệp video (*.mp4 *.avi)", options=options)
        if file_name:
            self.selected_image_file = file_name
            return self.selected_image_file
        else:
            return None

    def update_detected_objects_camera_video(self, detected_objects):
        self.detected_objects = detected_objects
        translated_objects = [self.class_name_map.get(obj, obj) for obj in detected_objects]
        unique_objects = set(translated_objects)
        detected_objects_str = '\n'.join(unique_objects)
        self.ui.ph_nt_textEdit.setPlainText(detected_objects_str)

    def update_detected_objects_images(self, detected_objects):
        self.detected_objects = detected_objects
        translated_objects = [self.class_name_map.get(obj, obj) for obj in detected_objects]
        object_counts = {}
        for obj in translated_objects:
            object_counts[obj] = object_counts.get(obj, 0) + 1
        detected_objects_str = ""
        for obj, count in object_counts.items():
            detected_objects_str += f"{obj}: {count}\n"
        self.ui.ph_nt_textEdit.setPlainText(detected_objects_str)

    # Chọn hình ảnh từ máy tính
    def original_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn hình ảnh", "", "Tệp hình ảnh (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            self.selected_image_file = file_name
            cv_img = cv2.imread(file_name)
            results = self.model(cv_img, conf=0.3, imgsz = 640)
            best_boxes = []
            detected_objects = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    cls = int(box.cls[0])
                    if self.classNames[cls] in ['book', 'clock', 'curtain', 'painting', 'vase', 'tv'] :
                        best_boxes.append(box)
                        detected_objects.append(self.classNames[cls])
                # Vẽ bounding box cho các hộp tốt nhất (nếu có)
                for best_box in best_boxes:
                    x1, y1, x2, y2 = best_box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # Vẽ bounding box
                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    # Tên lớp
                    cls = int(best_box.cls[0])
                    # Chi tiết vật thể
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(cv_img, self.classNames[cls], org, font, fontScale, color, thickness)
                qt_img = self.convert_cv_qt(cv_img)
                self.ui.original_label.setPixmap(qt_img)
                self.update_detected_objects_images(detected_objects)

    # Hiển thị thông tin về tệp được chọn
    def file_info(self):
        if self.selected_image_file:
            file_info = QFileInfo(self.selected_image_file)
            file_name = file_info.fileName()
            file_path = file_info.filePath()
            file_size = file_info.size()
            file_size_kb = file_size / 1024
            file_type = file_info.suffix()

            if file_type.lower() in ['mp4', 'avi']:
                vid = cv2.VideoCapture(self.selected_image_file)

                # Đọc thông tin video
                frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(vid.get(cv2.CAP_PROP_FPS))
                width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

                info_text = f"Tên tệp: {file_name}\n"
                info_text += f"Đường dẫn tệp: {file_path}\n"
                info_text += f"Kích thước tệp: {file_size_kb:.2f} KB\n"
                info_text += f"Loại tệp: {file_type.upper()}\n"
                info_text += f"Số khung hình: {frame_count}\n"
                info_text += f"FPS: {fps}\n"
                info_text += f"Chiều rộng: {width}\n"
                info_text += f"Chiều cao: {height}\n"
                self.ui.ttin_textEdit.setText(info_text)
                vid.release()
            else:
                info_text = f"Tên tệp: {file_name}\n"
                info_text += f"Đường dẫn tệp: {file_path}\n"
                info_text += f"Kích thước tệp: {file_size_kb:.2f} KB\n"
                info_text += f"Loại tệp: {file_type.upper()}\n"
                self.ui.ttin_textEdit.setText(info_text)

    # Xóa thông tin trên giao diện
    def clear_ui(self):
        self.stop_capture_video()
        self.ui.original_label.clear()
        self.ui.ttin_textEdit.clear()
        self.selected_image_file=''
        self.ui.ph_nt_textEdit.clear()
        self.ui.enter_textEdit.clear()
        self.ui.chat_bot_textEdit.clear()
        self.ui.gy_nt_textEdit.clear()

    def delete_chat(self):
        self.ui.chat_bot_textEdit.clear()

    # Xử lý sự kiện khi nhấn phím Enter trong enter_textEdit hoặc nhấn nút enter_pushButton
    def eventFilter(self, obj, event):
        if obj is self.ui.enter_textEdit and event.type() == QKeyEvent.KeyPress and event.key() == Qt.Key_Return:
            try:
                self.send_message()
            except Exception as e:
                print("Error:", e)
            return True
        return super().eventFilter(obj, event)

    # Gửi tin nhắn đến chatbot
    def send_message(self):
        try:
            msg = self.ui.enter_textEdit.toPlainText().strip()
            self.ui.enter_textEdit.clear()
            if msg:
                user_msg = "<img src='GUI_PyQt5/icon/user.jpg' width='50' height='50'>: " + msg.replace('\n','<br>') + '<br>'
                self.ui.chat_bot_textEdit.insertHtml(user_msg)

                res = chatbot_response(msg)
                bot_msg = "<img src='GUI_PyQt5/icon/chatbot.png' width='55' height='55'>: " + res.replace('\n','<br>') + '<br>'
                self.ui.chat_bot_textEdit.insertHtml(bot_msg)
        except Exception as e:
            print("Error:", e)

    # Hàm gợi ý
    def suggest_detected_objects(self):
        try:
            # Tính toán tập phổ biến
            frequent_itemsets = find_frequent_itemsets(extended_transactions(self.transactions, self.additional_items.values()), min_support=0.2)
            translated_objects = [self.class_name_map.get(obj, obj) for obj in self.detected_objects]
            input_items = set(translated_objects)
            suggestions = suggest_items(input_items, frequent_itemsets)
            translated_suggestions = []

            for suggestion in suggestions:
                translated_suggestion = []
                for obj in suggestion:
                    translated_obj = self.class_name_map.get(obj, obj)
                    translated_obj_additional = self.additional_items.get(obj, obj)
                    translated_suggestion.append(translated_obj if translated_obj != obj else translated_obj_additional)
                translated_suggestions.append(translated_suggestion)
            ht_suggestion = '\n'.join(', '.join(suggestion) for suggestion in translated_suggestions)

            self.ui.gy_nt_textEdit.setPlainText(ht_suggestion)

        except Exception as e:
            print("Error occurred:", e)
            error_msg = traceback.format_exc()
            print(error_msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
