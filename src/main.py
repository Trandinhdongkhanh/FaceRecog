import argparse
import collections
import pickle
import tkinter as tk
import os
from tkinter import filedialog

import imutils
from PIL import ImageTk, Image
import PIL.Image, PIL.ImageTk
from tkinter.scrolledtext import ScrolledText

from imutils.video import VideoStream

import face_rec_cam as camReg
import face_rec as videoReg
import cv2
import tensorflow as tf
import numpy as np

import align.detect_face
import facenet

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
PIC_WIDTH = 600
PIC_HEIGHT = 450
BTN_WIDTH = 12
BTN_HEIGHT = 2
LOGO_WIDTH = 100


class Window(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Face Recognition')
        self.geometry('900x500')
        self.resizable(False, False)


from imutils.video import VideoStream


class App():
    def __init__(self):
        self.win = Window()
        self.upFrame = tk.Frame(self.win, height=500, width=900, bg='#7FB3D5')
        self.lbl_Pic = tk.Label(self.upFrame)

        logo = ImageTk.PhotoImage(
            Image.open('Images/logo_img.jpg').resize((WINDOW_WIDTH - (PIC_WIDTH + 40), 100), Image.ANTIALIAS))
        self.pic_logo = tk.Label(self.upFrame, image=logo)
        self.pic_logo.image = logo

        self.btnImageDetect = tk.Button(self.upFrame, width=BTN_WIDTH, height=BTN_HEIGHT, text='Image Detect',
                                        command=self.img_detect)
        self.btnClear = tk.Button(self.upFrame, width=BTN_WIDTH, height=BTN_HEIGHT, text='Clear',
                                  command=self.clear)
        self.btnCamDetect = tk.Button(self.upFrame, width=BTN_WIDTH, height=BTN_HEIGHT, text='Camera Detect',
                                      command=camReg.main)
        self.btnVideoDetect = tk.Button(self.upFrame, width=BTN_WIDTH, height=BTN_HEIGHT, text='Video Detect',
                                        command=videoReg.main)

        self.upFrame.pack()

        self.lbl_Pic.place(x=20, y=20, width=PIC_WIDTH, height=PIC_HEIGHT)
        self.pic_logo.place(x=PIC_WIDTH + 30, y=20, width=WINDOW_WIDTH - (PIC_WIDTH + 40), height=100)

        self.btnImageDetect.place(x=670, y=160)
        self.btnClear.place(x=770, y=160)
        self.btnCamDetect.place(x=670, y=220)
        self.btnVideoDetect.place(x=770, y=220)

    def clear(self):
        self.lbl_Pic.configure(image=None)
        self.lbl_Pic.image = None

    def img_detect(self):
        file_path = FileBrowser.browseFile()
        if file_path == '':  # If browseFile() return an empty string on cancel
            return
        print(file_path)
        parser = argparse.ArgumentParser()
        parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
        args = parser.parse_args()

        MINSIZE = 20
        THRESHOLD = [0.6, 0.7, 0.7]
        FACTOR = 0.709
        IMAGE_SIZE = 182
        INPUT_IMAGE_SIZE = 160
        CLASSIFIER_PATH = 'Models/facemodel.pkl'
        VIDEO_PATH = args.path
        FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

        # Load The Custom Classifier
        with open(CLASSIFIER_PATH, 'rb') as file:
            model, class_names = pickle.load(file)
        print("Custom Classifier, Successfully loaded")

        with tf.Graph().as_default():

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

            with sess.as_default():

                # Load the model
                print('Loading feature extraction model')
                facenet.load_model(FACENET_MODEL_PATH)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

                people_detected = set()
                person_detected = collections.Counter()

                # Load image from file
                img = cv2.imread(file_path)

                # Resize image
                resized = cv2.resize(img, (0, 0), fx=1, fy=1)

                # Assign the resized image to frame
                frame = resized

                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            print(bb[i][3] - bb[i][1])
                            print(frame.shape[0])
                            print((bb[i][3] - bb[i][1]) / frame.shape[0])
                            if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                                # Ve khung mau xanh quanh khuon mat
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20

                                # Neu ty le nhan dang > 0.5 thi hien thi ten
                                if best_class_probabilities > 0.5:
                                    name = class_names[best_class_indices[0]]
                                else:
                                    # Con neu <=0.5 thi hien thi Unknow
                                    name = "Unknown"

                                # Viet text len tren frame
                                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                                cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                                person_detected[best_name] += 1

                                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # Convert frame to RGB format
                                img = Image.fromarray(cv2image)  # Convert numpy array to Image object
                                img = img.resize((PIC_WIDTH, PIC_HEIGHT), Image.ANTIALIAS)  # Resize image to fit label
                                input_img = ImageTk.PhotoImage(img)
                                self.lbl_Pic.configure(image=input_img)  # Update label with new image
                                self.lbl_Pic.image = input_img

                                cv2.imshow("Face Recognition", frame)
                                cv2.waitKey(0)

                except:
                    pass


class FileBrowser():
    def __init__(self):
        pass

    @staticmethod
    def browseFile():
        currdir = os.getcwd()  # current python directory when open filedialog
        filename = filedialog.askopenfilename(initialdir=currdir, title='Please select an image', filetypes=[
            ("image", ".jpeg"),
            ("image", ".png"),
            ("image", ".jpg"),
            ("image", ".bmp")
        ])
        return filename


app = App()
app.win.mainloop()
