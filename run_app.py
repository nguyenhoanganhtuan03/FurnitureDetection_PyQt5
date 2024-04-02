import numpy as np
import sys
import os
import cv2

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap

from GUI_PyQt5.app_ui import Ui_MainWindow  # Import your generated UI module

def video_or_camera():
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file_name, _ = QFileDialog.getOpenFileName(None, "Select Video File", "", "Video Files (*.mp4 *.avi)",
                                               options=options)
    if file_name:
        return file_name
    else:
        return 0

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # self.ui.ha_pushButton.clicked.connect(self.original_image)
        self.ui.video_pushButton.clicked.connect(self.start_capture_video)
        self.ui.camera_pushButton.clicked.connect(self.start_capture_video)
        self.ui.stop_pushButton.clicked.connect(self.stop_capture_video)

        self.thread = {}

    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        try:
            if 1 in self.thread:
                self.thread[1].stop()
                self.thread[1].wait()
                self.ui.original_label.clear()
        except Exception as e:
            print("Error stopping video capture thread:", e)

    def start_capture_video(self):
        try:
            if 1 in self.thread:
                QMessageBox.warning(self, "Warning", "Video capture is already running.")
                return
            self.thread[1] = capture_video(index=1)
            self.thread[1].start()
            self.thread[1].signal.connect(self.show_webcam)
        except Exception as e:
            print("Error starting video capture thread:", e)

    def show_webcam(self, cv_img):
        """Updates the original_label with a new OpenCV image"""
        try:
            qt_img = self.convert_cv_qt(cv_img)
            self.ui.original_label.setPixmap(qt_img)
        except Exception as e:
            print("Error displaying webcam image:", e)

    def convert_cv_qt(self, cv_img):
        """Convert from an OpenCV image to QPixmap"""
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(990, 660, Qt.KeepAspectRatio)
            return QPixmap.fromImage(p)
        except Exception as e:
            print("Error converting OpenCV image to QPixmap:", e)

    # def original_image(self):


class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, index):
        self.index = index
        print("Start Threading", self.index)
        super(capture_video, self).__init__()

    def run(self):
        try:
            cap = cv2.VideoCapture(video_or_camera())
            if not cap.isOpened():
                raise Exception("Camera could not be opened")

            self.is_running = True
            while True:
                ret, cv_img = cap.read()
                if ret:
                    self.signal.emit(cv_img)
        except Exception as e:
            print("Error in video capture thread:", e)

    def stop(self):
        print("Stop Threading", self.index)
        self.is_running = False
        self.terminate()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
