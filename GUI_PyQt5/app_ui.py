# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 170, 291, 351))
        self.widget.setStyleSheet("background-color: rgb(87, 247, 255);\n"
"border-radius: 40px;\n"
"border: 2px solid blue;\n"
"")
        self.widget.setObjectName("widget")
        self.ha_pushButton = QtWidgets.QPushButton(self.widget)
        self.ha_pushButton.setGeometry(QtCore.QRect(70, 70, 151, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.ha_pushButton.setFont(font)
        self.ha_pushButton.setStyleSheet("QPushButton {\n"
"    background-color: rgb(255, 255, 255);\n"
"    border-radius: 10px;\n"
"    transition: background-color 0.3s, transform 0.3s;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"       background-color: rgb(209, 209, 209);\n"
"}\n"
"")
        self.ha_pushButton.setObjectName("ha_pushButton")
        self.camera_pushButton = QtWidgets.QPushButton(self.widget)
        self.camera_pushButton.setGeometry(QtCore.QRect(70, 210, 151, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.camera_pushButton.setFont(font)
        self.camera_pushButton.setStyleSheet("QPushButton {\n"
"    background-color: rgb(255, 255, 255);\n"
"    border-radius: 10px;\n"
"    transition: background-color 0.3s, transform 0.3s;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"       background-color: rgb(209, 209, 209);\n"
"}\n"
"")
        self.camera_pushButton.setObjectName("camera_pushButton")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(10, 20, 271, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("border: none;\n"
"")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.stop_pushButton = QtWidgets.QPushButton(self.widget)
        self.stop_pushButton.setGeometry(QtCore.QRect(70, 280, 151, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.stop_pushButton.setFont(font)
        self.stop_pushButton.setStyleSheet("QPushButton {\n"
"    background-color: rgb(255, 255, 255);\n"
"    border-radius: 10px;\n"
"    transition: background-color 0.3s, transform 0.3s;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"       background-color: rgb(209, 209, 209);\n"
"}\n"
"")
        self.stop_pushButton.setObjectName("stop_pushButton")
        self.video_pushButton = QtWidgets.QPushButton(self.widget)
        self.video_pushButton.setGeometry(QtCore.QRect(70, 140, 151, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.video_pushButton.setFont(font)
        self.video_pushButton.setStyleSheet("QPushButton {\n"
"    background-color: rgb(255, 255, 255);\n"
"    border-radius: 10px;\n"
"    transition: background-color 0.3s, transform 0.3s;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"       background-color: rgb(209, 209, 209);\n"
"}\n"
"")
        self.video_pushButton.setObjectName("video_pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(6, 3, 1591, 161))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(28)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.widget_3 = QtWidgets.QWidget(self.centralwidget)
        self.widget_3.setGeometry(QtCore.QRect(10, 530, 291, 321))
        self.widget_3.setStyleSheet("background-color: rgb(87, 247, 255);\n"
"border-radius: 40px;\n"
"border: 2px solid blue;")
        self.widget_3.setObjectName("widget_3")
        self.label_3 = QtWidgets.QLabel(self.widget_3)
        self.label_3.setGeometry(QtCore.QRect(20, 10, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("border: none;\n"
"")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.ttin_textEdit = QtWidgets.QTextEdit(self.widget_3)
        self.ttin_textEdit.setGeometry(QtCore.QRect(10, 60, 271, 251))
        self.ttin_textEdit.setObjectName("ttin_textEdit")
        self.widget_4 = QtWidgets.QWidget(self.centralwidget)
        self.widget_4.setGeometry(QtCore.QRect(1300, 170, 291, 341))
        self.widget_4.setStyleSheet("background-color: rgb(87, 247, 255);\n"
"border-radius: 40px;\n"
"border: 2px solid blue;")
        self.widget_4.setObjectName("widget_4")
        self.label_4 = QtWidgets.QLabel(self.widget_4)
        self.label_4.setGeometry(QtCore.QRect(20, 10, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("border: none;\n"
"")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.ph_nt_textEdit = QtWidgets.QTextEdit(self.widget_4)
        self.ph_nt_textEdit.setGeometry(QtCore.QRect(10, 60, 271, 271))
        self.ph_nt_textEdit.setObjectName("ph_nt_textEdit")
        self.widget_5 = QtWidgets.QWidget(self.centralwidget)
        self.widget_5.setGeometry(QtCore.QRect(1300, 520, 291, 331))
        self.widget_5.setStyleSheet("background-color: rgb(87, 247, 255);\n"
"border-radius: 40px;\n"
"border: 2px solid blue;")
        self.widget_5.setObjectName("widget_5")
        self.gy_nt_pushButton = QtWidgets.QPushButton(self.widget_5)
        self.gy_nt_pushButton.setGeometry(QtCore.QRect(40, 10, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.gy_nt_pushButton.setFont(font)
        self.gy_nt_pushButton.setStyleSheet("QPushButton {\n"
"    background-color:rgb(66, 255, 126);\n"
"    border-radius: 10px;\n"
"    border: none;\n"
"    transition: background-color 0.3s, transform 0.3s;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"       background-color: rgb(232, 255, 25);\n"
"}")
        self.gy_nt_pushButton.setObjectName("gy_nt_pushButton")
        self.gy_nt_textEdit = QtWidgets.QTextEdit(self.widget_5)
        self.gy_nt_textEdit.setGeometry(QtCore.QRect(10, 70, 271, 251))
        self.gy_nt_textEdit.setObjectName("gy_nt_textEdit")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(310, 170, 981, 681))
        self.tabWidget.setStyleSheet("")
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.original_label = QtWidgets.QLabel(self.tab)
        self.original_label.setGeometry(QtCore.QRect(0, 0, 981, 661))
        self.original_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"")
        self.original_label.setText("")
        self.original_label.setObjectName("original_label")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.reco_label = QtWidgets.QLabel(self.tab_2)
        self.reco_label.setGeometry(QtCore.QRect(-10, -10, 991, 671))
        self.reco_label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"")
        self.reco_label.setText("")
        self.reco_label.setObjectName("reco_label")
        self.tabWidget.addTab(self.tab_2, "")
        self.label.raise_()
        self.widget.raise_()
        self.widget_3.raise_()
        self.widget_4.raise_()
        self.widget_5.raise_()
        self.tabWidget.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ha_pushButton.setText(_translate("MainWindow", "Hình Ảnh"))
        self.camera_pushButton.setText(_translate("MainWindow", "Camera"))
        self.label_2.setText(_translate("MainWindow", "Upload"))
        self.stop_pushButton.setText(_translate("MainWindow", "Stop"))
        self.video_pushButton.setText(_translate("MainWindow", "Video"))
        self.label.setText(_translate("MainWindow", "Hệ thống nhận dạng và gợi ý nội thất phòng khách"))
        self.label_3.setText(_translate("MainWindow", "Thông tin File"))
        self.label_4.setText(_translate("MainWindow", "Nhận dạng"))
        self.gy_nt_pushButton.setText(_translate("MainWindow", "Gợi ý nội thất"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Original file"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Recognized file"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
