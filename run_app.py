import numpy as np
import sys
import cv2
from ultralytics import YOLO

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QFileInfo
from PyQt5.QtGui import QPixmap
from GUI_PyQt5.app_ui import Ui_MainWindow

from mlxtend.frequent_patterns import apriori
from apriori_nt import transactions

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
        detected_objects = []
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
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            confidence = box.conf[0]
                            cls = int(box.cls[0])
                            if self.classNames[cls] in ['book', 'clock', 'curtain', 'painting', 'vase', 'tv']:
                                best_boxes.append(box)
                                detected_objects.append(self.classNames[cls])
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
                else:
                    break
        except Exception as e:
            print("Lỗi trong luồng chụp video:", e)
        finally:
            if self.keep_running:
                self.finished_signal.emit(detected_objects)

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

        # Kết nối các sự kiện với các hàm tương ứng
        self.ui.ha_pushButton.clicked.connect(self.original_image)
        self.ui.ha_pushButton.clicked.connect(self.file_info)
        self.ui.video_pushButton.clicked.connect(self.start_capture_video)
        self.ui.video_pushButton.clicked.connect(self.file_info)
        self.ui.camera_pushButton.clicked.connect(self.start_capture_camera)
        self.ui.stop_pushButton.clicked.connect(self.stop_capture_video)
        self.ui.clear_pushButton.clicked.connect(self.clear_ui)
        self.ui.gy_nt_pushButton.clicked.connect(self.suggest_detected_objects)

    # Xử lý sự kiện khi cửa sổ đóng
    def closeEvent(self, event):
        self.stop_capture_video()

    # Dừng chụp video
    def stop_capture_video(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None
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

    # Hàm hiển thị gợi ý
    def suggest_detected_objects(self, transactions):
        frequent_itemsets = apriori(transactions, min_support=0.2, use_colnames=True)
        frequent_itemsets_str = frequent_itemsets.to_string()
        self.ui.gy_nt_textEdit.setPlainText(frequent_itemsets_str)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
