from turtle import Screen
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2, time
import numpy as np
from numba import njit, prange

def conv_cv_to_qpixmap(img_cv):
    height, width, channel = img_cv.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(img_cv.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
    return QtGui.QPixmap(qImg)

@njit(fastmath = True, parallel = True)
def gen_gaussian_kernel_numba_2d(kernel_size, sigma):
    if sigma == 0: sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8
    kernelRadius = int((kernel_size - 1) / 2)
    karnel = np.zeros((kernel_size, kernel_size))
    karnel_sum = 0
    for i in prange(kernel_size): 
        for j in prange(kernel_size): 
            x = np.exp(-((i - kernelRadius)**2 + (j - kernelRadius)**2)/(2*sigma**2)) / (2*np.pi*sigma**2)
            karnel[i][j] = x
            karnel_sum += x
    for i in prange(kernel_size):
        for j in prange(kernel_size):
            karnel[i][j] = karnel[i][j] / karnel_sum
    return karnel

@njit(fastmath = True)
def gen_gaussian_kernel_numba(kernel_size, sigma):
    if sigma == 0: sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8
    kernelRadius = int((kernel_size - 1) / 2)
    karnel = np.zeros(kernel_size)
    karnel_sum = 0
    for i in prange(kernel_size): 
        # x = np.exp(- (i - kernelRadius) ** 2 / (2 * sigma ** 2))
        x = np.exp(-((i - kernelRadius)**2)/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
        karnel[i] = x
        karnel_sum += x
    for i in prange(kernel_size):
        karnel[i] = karnel[i] / karnel_sum
    return karnel

def gen_gaussian_kernel(kernel_size, sigma):
    if sigma == 0: sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8
    kernelRadius = int((kernel_size - 1) / 2)
    # karnel = np.array([np.exp(- (i - kernelRadius) ** 2 / (2 * sigma ** 2)) for i in prange(kernel_size)])   
    karnel = np.array([np.exp(-((i - kernelRadius)**2)/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2) for i in prange(kernel_size)])   
    karnel = karnel / karnel.sum() 
    return karnel

def GaussianBlur(img_cv, kernel_size, sigma):
    img = img_cv.copy()
    karnel = gen_gaussian_kernel(kernel_size, sigma)
    kernelRadius = int((kernel_size - 1) / 2)
    height, width, channel = img.shape
    for channel_i in range(channel):
        for width_i in range(width):
            for height_i in range(height):
                sum_pix = 0
                for i in range(kernel_size):
                    h = height_i - kernelRadius + i 
                    sum_pix += img_cv[abs(h) if h < height else 2*height - h - 1][width_i][channel_i] * karnel[i] 
                img[height_i][width_i][channel_i] = sum_pix
                
        for width_i in range(width):
            for height_i in range(height):
                sum_pix = 0
                for j in range(kernel_size):
                    w = width_i - kernelRadius + j 
                    sum_pix += img[height_i][abs(w) if w < width else 2*width - w - 1][channel_i] * karnel[j]
                img[height_i][width_i][channel_i] = sum_pix
    return img

@njit(fastmath = True, parallel = True)
def GaussianBlur_numba(img_cv, kernel_size, sigma):
    img = img_cv.copy()
    kernelRadius = int((kernel_size - 1) / 2)
    karnel = gen_gaussian_kernel_numba(kernel_size, sigma)
    # print(karnel)
    # karnel = cv2.getGaussianKernel(kernel_size, sigma))
    # karnel = np.array([0,0,0,0,0,1,0,0,0,0,0])
    height, width, channel = img.shape
    for channel_i in prange(channel):
        for width_i in range(width):
            for height_i in range(height):
                sum_pix = 0
                for i in range(kernel_size):
                    h = height_i - kernelRadius + i 
                    sum_pix += img_cv[abs(h) if h < height else 2*height - h - 1][width_i][channel_i] * karnel[i] 
                img[height_i][width_i][channel_i] = sum_pix
                
        for width_i in range(width):
            for height_i in range(height):
                sum_pix = 0
                for j in range(kernel_size):
                    w = width_i - kernelRadius + j 
                    sum_pix += img[height_i][abs(w) if w < width else 2*width - w - 1][channel_i] * karnel[j]
                img[height_i][width_i][channel_i] = sum_pix
    return img
    
@njit(fastmath = True, parallel = True)
def GaussianBlur_numba_2d(img_cv, kernel_size, sigma):
    img = img_cv.copy()
    kernelRadius = int((kernel_size - 1) / 2)
    karnel = gen_gaussian_kernel_numba_2d(kernel_size, sigma)
    height, width, channel = img.shape
    for channel_i in prange(channel):
        for width_i in range(width):
            for height_i in range(height):
                sum_pix = 0
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        h = height_i - kernelRadius + i 
                        w = width_i - kernelRadius + j 
                        sum_pix += img[abs(h) if h < height else 2*height - h - 1][abs(w) if w < width else 2*width - w - 1][channel_i] * karnel[i][j] 
                img[height_i][width_i][channel_i] = sum_pix
    return img

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(566, 433)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.images = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.images.sizePolicy().hasHeightForWidth())
        self.images.setSizePolicy(sizePolicy)
        self.images.setScaledContents(True)
        self.images.setObjectName("images")
        self.verticalLayout.addWidget(self.images)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 1, 1, 1, 1)
        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_1.setObjectName("pushButton_1")
        self.gridLayout.addWidget(self.pushButton_1, 1, 2, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 0, 1, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout.addWidget(self.comboBox, 0, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.images.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_2.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_1.setText(_translate("MainWindow", "PushButton"))
        self.spinBox.setValue(5)
        self.pushButton_2.setText('Processed')
        self.pushButton_1.setText('Original')
        self.comboBox.addItem('CV2')
        self.comboBox.addItem('Python')
        self.comboBox.addItem('Numba')
        self.comboBox.addItem('Numba_2d')
        self.pushButton_1.clicked.connect(self.get_org_img)
        self.pushButton_2.clicked.connect(self.get_red_img)
        self.img_cv = cv2.imread('img/2.png')   
        self.images.setPixmap(conv_cv_to_qpixmap(self.img_cv))  
        self.spinBox.setSingleStep(2)
        GaussianBlur_numba_2d(self.img_cv, 1, 1)
        GaussianBlur_numba(self.img_cv, 1, 1)
        
    def get_org_img(self):     
        self.images.setPixmap(conv_cv_to_qpixmap(self.img_cv))  
        
    def get_red_img(self):         
        karel_size = self.spinBox.value()
        sigma = 0.3*((karel_size-1)*0.5 - 1) + 0.8
        if karel_size % 2 == 1:
            self.statusbar.showMessage('Processing...') 
            start_time = time.time()   
            if self.comboBox.currentText() == 'CV2':
                self.images.setPixmap(conv_cv_to_qpixmap(cv2.GaussianBlur(self.img_cv, (karel_size, karel_size),sigma)))  
            elif self.comboBox.currentText() == 'Python':
                self.images.setPixmap(conv_cv_to_qpixmap(GaussianBlur(self.img_cv, karel_size, sigma)))
            elif self.comboBox.currentText() == 'Numba':
                self.images.setPixmap(conv_cv_to_qpixmap(GaussianBlur_numba(self.img_cv, karel_size, sigma)))
                # cv2.imwrite( "screen\\5.png", GaussianBlur_numba(self.img_cv, karel_size, sigma))  
            else: 
                self.images.setPixmap(conv_cv_to_qpixmap(GaussianBlur_numba_2d(self.img_cv, karel_size, sigma)))
                # self.images.setPixmap(conv_cv_to_qpixmap(cv2.filter2D(self.img_cv, -1, gen_gaussian_kernel_numba_2d(karel_size, sigma))))
            self.statusbar.showMessage("--- %s seconds ---" % (time.time() - start_time))   
              
    
                        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())