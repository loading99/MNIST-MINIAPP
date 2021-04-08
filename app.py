import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from functools import partial
import Digit
#tensorflow and keras
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model




#file uploaded needs to be 28x28
def upload_file(ui):

    loadImg(ui)
    im_matrix=convert(ui.file)
    print(im_matrix.shape)

    #Make prediction
    clf=load_model("MNIST")
    prediction=clf.predict_classes(im_matrix)[0]
    print(prediction)
    ui.lineEdit.setText(str(prediction))


# load and prepare the image
def convert(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

def loadImg(ui):
    print("--------uploading-----")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    print(file_path)
    ui.file=file_path

if __name__ == '__main__':
    app=QApplication(sys.argv)
    MainWindow=QMainWindow()
    ui=Digit.Ui_MainWindow()
    ui.setupUi(MainWindow)
    print("--------starting-----")

    ui.upload.clicked.connect(partial(upload_file,ui))
    MainWindow.show()
    sys.exit(app.exec_())



