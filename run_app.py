import numpy as np
import sys
import os
import cv2
from ultralytics import YOLO

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QFileInfo
from PyQt5.QtGui import QPixmap

from GUI_PyQt5.app_ui import Ui_MainWindow

# Định nghĩa một lớp con QThread để chụp video
import numpy as np
import cv2
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal

class CaptureVideo(QThread):
    signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal()  # Thêm tín hiệu để thông báo khi video kết thúc

    def __init__(self, file_path):
        self.file_path = file_path
        self.cap = None
        self.model = YOLO("Model_Yolo/src_code/runs/detect/train/weights/best.pt")
        self.classNames = ['book', 'clock', 'curtain', 'painting', 'vase', 'tv']
        super().__init__()

    # Hàm run để bắt đầu chụp video
    def run(self):
        detected_objects = []
        try:
            if self.file_path is not None:
                self.cap = cv2.VideoCapture(self.file_path)
            else:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Không thể mở camera")
            while True:
                ret, img = self.cap.read()
                if ret:
                    results = self.model(img, stream=True)

                    # Xử lý kết quả
                    max_confidence = 0.3
                    best_boxes = []

                    for r in results:
                        boxes = r.boxes

                        for box in boxes:
                            confidence = box.conf[0]
                            cls = int(box.cls[0])
                            if self.classNames[cls] in ['book', 'clock', 'curtain', 'painting', 'vase', 'tv'] and confidence > max_confidence:
                                max_confidence = confidence
                                best_boxes.append(box)
                                detected_objects.append(self.classNames[cls])

                    # Vẽ bounding box cho các hộp tốt nhất (nếu có)
                    for best_box in best_boxes:
                        x1, y1, x2, y2 = best_box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Vẽ bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # Tên lớp
                        cls = int(best_box.cls[0])

                        # Chi tiết vật thể
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(img, self.classNames[cls], org, font, fontScale, color, thickness)
                    self.signal.emit(img)
                else:
                    break
            self.finished_signal.emit()  # Gửi tín hiệu khi video kết thúc
        except Exception as e:
            print("Lỗi trong luồng chụp video:", e)

    # Hàm stop để dừng chụp video
    def stop(self):
        self.terminate()


# Định nghĩa lớp MainWindow để quản lý giao diện chính
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.selected_image_file = ""
        self.thread = None
        self.model = YOLO("Model_Yolo/src_code/runs/detect/train/weights/best.pt")
        self.classNames = ['book', 'clock', 'curtain', 'painting', 'vase', 'tv']

        # Kết nối các sự kiện với các hàm tương ứng
        self.ui.ha_pushButton.clicked.connect(self.original_image)
        self.ui.ha_pushButton.clicked.connect(self.file_info)
        self.ui.ha_pushButton.clicked.connect(self.load_image)
        self.ui.video_pushButton.clicked.connect(self.start_capture_video)
        self.ui.video_pushButton.clicked.connect(self.file_info)
        self.ui.camera_pushButton.clicked.connect(self.start_capture_camera)
        self.ui.stop_pushButton.clicked.connect(self.stop_capture_video)
        self.ui.clear_pushButton.clicked.connect(self.clear_ui)

    # Xử lý sự kiện khi cửa sổ đóng
    def closeEvent(self, event):
        self.stop_capture_video()

    # Dừng chụp video
    def stop_capture_video(self):
        try:
            if self.thread:
                self.thread.stop()
                self.thread.wait()
                self.ui.original_label.clear()
                self.thread = None
        except Exception as e:
            print("Lỗi khi dừng luồng chụp video:", e)

    # Bắt đầu chụp video từ tệp đã chọn
    def start_capture_video(self):
        try:
            if not self.thread:
                file_path = self.video_or_camera()
                if file_path:
                    self.thread = CaptureVideo(file_path)
                    self.thread.start()
                    self.thread.signal.connect(self.show_webcam)
        except Exception as e:
            print("Lỗi khi bắt đầu luồng chụp video:", e)

    # Bắt đầu chụp video từ camera
    def start_capture_camera(self):
        try:
            if not self.thread:
                self.thread = CaptureVideo(None)
                self.thread.start()
                self.thread.signal.connect(self.show_webcam)
        except Exception as e:
            print("Lỗi khi bắt đầu luồng chụp camera:", e)

    # Hiển thị hình ảnh từ webcam
    def show_webcam(self, cv_img):
        try:
            qt_img = self.convert_cv_qt(cv_img)
            self.ui.original_label.setPixmap(qt_img)
        except Exception as e:
            print("Lỗi khi hiển thị hình ảnh webcam:", e)

    # Chuyển đổi hình ảnh từ OpenCV sang QPixmap
    def convert_cv_qt(self, cv_img):
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(990, 660, Qt.KeepAspectRatio)
            return QPixmap.fromImage(p)
        except Exception as e:
            print("Lỗi khi chuyển đổi hình ảnh OpenCV thành QPixmap:", e)

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

    # Chọn hình ảnh từ máy tính
    def original_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn hình ảnh", "","Tệp hình ảnh (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            self.selected_image_file = file_name
            try:
                cv_img = cv2.imread(file_name)
                if cv_img is not None:
                    qt_img = self.convert_cv_qt(cv_img)
                    self.ui.original_label.setPixmap(qt_img)
                else:
                    QMessageBox.warning(self, "Cảnh báo", "Tệp đã chọn không phải là hình ảnh hợp lệ.")
            except Exception as e:
                print("Lỗi khi tải hình ảnh:", e)
        else:
            QMessageBox.warning(self, "Cảnh báo", "Không có hình ảnh nào được chọn.")

    # Hiển thị thông tin về tệp được chọn
    def file_info(self):
        if self.selected_image_file:
            try:
                file_info = QFileInfo(self.selected_image_file)
                file_name = file_info.fileName()
                file_path = file_info.filePath()
                file_size = file_info.size()
                file_size_kb = file_size / 1024
                file_type = file_info.suffix()

                if file_type.lower() in ['mp4', 'avi']:
                    vid = cv2.VideoCapture(self.selected_image_file)
                    if not vid.isOpened():
                        QMessageBox.warning(self, "Cảnh báo", "Không thể mở tệp video.")
                        return

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
            except Exception as e:
                print("Lỗi trong thông tin tệp:", e)
                QMessageBox.warning(self, "Cảnh báo", "Đã xảy ra lỗi trong quá trình xử lý tệp.")
        else:
            QMessageBox.warning(self, "Cảnh báo", "Không có tệp nào được chọn.")

    # Xóa thông tin trên giao diện
    def clear_ui(self):
        self.stop_capture_video()
        self.ui.original_label.clear()
        self.ui.ttin_textEdit.clear()
        self.selected_image_file=''

    # Tải hình ảnh và phát hiện vật thể
    def load_image(self):
        detected_objects = []
        try:
            if self.selected_image_file:
                img = cv2.imread(self.selected_image_file)
                if img is not None:
                    results = self.model(img, conf=0.3, imgsz = 640)
                    for obj in results:
                        detected_objects.append(obj['label'])
        except Exception as e:
            QMessageBox.warning(self, "Cảnh báo", "Đã xảy ra lỗi khi tải mô hình.")
        self.ph_nt_textEdit.setText(detected_objects)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
