import numpy as np
import sys
import os
import cv2

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QFileInfo
from PyQt5.QtGui import QPixmap

from GUI_PyQt5.app_ui import Ui_MainWindow  # Import your generated UI module

class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, file_path):
        self.file_path = file_path
        self.cap = None  # Khởi tạo biến cap
        super().__init__()

    def run(self):
        try:
            if self.file_path is not None:  # Kiểm tra nếu file_path không phải là None
                self.cap = cv2.VideoCapture(self.file_path)
            else:  # Trường hợp sử dụng camera
                self.cap = cv2.VideoCapture(0)  # Sử dụng camera với index 0
                if not self.cap.isOpened():  # Kiểm tra xem camera có được mở hay không
                    raise Exception("Camera could not be opened")
            while True:
                ret, cv_img = self.cap.read()
                if ret:
                    self.signal.emit(cv_img)
                else:
                    break
        except Exception as e:
            print("Error in video capture thread:", e)
        finally:
            if self.cap is not None:
                self.cap.release()  # Giải phóng camera

    def stop(self):
        self.terminate()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.selected_image_file = ""
        self.thread = None  # Khởi tạo thread

        self.ui.ha_pushButton.clicked.connect(self.original_image)
        self.ui.ha_pushButton.clicked.connect(self.file_info)
        self.ui.video_pushButton.clicked.connect(self.start_capture_video)
        self.ui.camera_pushButton.clicked.connect(self.start_capture_camera)
        self.ui.stop_pushButton.clicked.connect(self.stop_capture_video)
        self.ui.clear_pushButton.clicked.connect(self.clear_ui)

    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        try:
            if self.thread:  # Kiểm tra xem thread đã được khởi tạo chưa
                self.thread.stop()  # Dừng thread
                self.thread.wait()  # Chờ thread kết thúc
                self.ui.original_label.clear()
                self.thread = None  # Đặt lại thread thành None
        except Exception as e:
            print("Error stopping video capture thread:", e)

    def start_capture_video(self):
        try:
            if not self.thread:
                file_path = self.video_or_camera()
                if file_path:
                    self.thread = capture_video(file_path)
                    self.thread.start()
                    self.thread.signal.connect(self.show_webcam)
        except Exception as e:
            print("Error starting video capture thread:", e)

    def start_capture_camera(self):
        try:
            if not self.thread:
                self.thread = capture_video(None)  # Truyền None cho đường dẫn camera
                self.thread.start()
                self.thread.signal.connect(self.show_webcam)
        except Exception as e:
            print("Error starting camera capture thread:", e)

    def show_webcam(self, cv_img):
        try:
            qt_img = self.convert_cv_qt(cv_img)
            self.ui.original_label.setPixmap(qt_img)
        except Exception as e:
            print("Error displaying webcam image:", e)

    def convert_cv_qt(self, cv_img):
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(990, 660, Qt.KeepAspectRatio)
            return QPixmap.fromImage(p)
        except Exception as e:
            print("Error converting OpenCV image to QPixmap:", e)

    def video_or_camera(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)", options=options)
        if file_name:
            self.selected_image_file = file_name
            return self.selected_image_file
        else:
            return None

    def original_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "","Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            self.selected_image_file = file_name
            try:
                cv_img = cv2.imread(file_name)
                if cv_img is not None:
                    qt_img = self.convert_cv_qt(cv_img)
                    self.ui.original_label.setPixmap(qt_img)
                else:
                    QMessageBox.warning(self, "Warning", "Selected file is not a valid image.")
            except Exception as e:
                print("Error loading image:", e)
        else:
            QMessageBox.warning(self, "Warning", "No image selected.")

    def file_info(self):
        if self.selected_image_file:
            try:
                file_info = QFileInfo(self.selected_image_file)
                file_name = file_info.fileName()
                file_path = file_info.filePath()
                file_size = file_info.size()
                file_size_kb = file_size / 1024
                file_type = file_info.suffix()

                if file_type.lower() in ['.mp4', '.avi']:
                    cap = cv2.VideoCapture(self.selected_image_file)
                    if not cap.isOpened():
                        QMessageBox.warning(self, "Warning", "Cannot open video file.")
                        return

                    # Đọc thông tin video
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    info_text = "\n"
                    info_text += f"File Name: {file_name}\n"
                    info_text += f"File Path: {file_path}\n"
                    info_text += f"File Size: {file_size_kb:.2f} KB\n"
                    info_text += f"File Type: {file_type.upper()}\n"
                    info_text += f"Frame Count: {frame_count}\n"
                    info_text += f"FPS: {fps}\n"
                    info_text += f"Width: {width}\n"
                    info_text += f"Height: {height}\n"

                    self.ui.ttin_textEdit.setText(info_text)
                    print(fps)

                    cap.release()  # Giải phóng tài nguyên video
                else:
                    info_text = "\n"
                    info_text += f"File Name: {file_name}\n"
                    info_text += f"File Path: {file_path}\n"
                    info_text += f"File Size: {file_size_kb:.2f} KB\n"
                    info_text += f"File Type: {file_type.upper()}\n"

                    self.ui.ttin_textEdit.setText(info_text)
            except Exception as e:
                print("Error in file_info:", e)
                QMessageBox.warning(self, "Warning", "An error occurred while processing the file.")
        else:
            QMessageBox.warning(self, "Warning", "No file selected.")

    def clear_ui(self):
        self.stop_capture_video()
        self.ui.original_label.clear()
        self.ui.ttin_textEdit.clear()
        self.selected_image_file=''


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
