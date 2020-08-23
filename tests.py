from unittest import TestCase
import cv2
from tensorflow.keras.models import Sequential
from drowsiness_detection import detect_face, cnn_preprocess, build_model, crop_eyes


class Test(TestCase):

    def test_detect_face(self):
        face = cv2.imread('test_imgs/face.jpg')

        # convert image to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # detect the face at grayscale image
        detected = detect_face(gray, minimum_feature_size=(80, 80))

        self.assertTrue(detected != [])

    def test_crop_eyes(self):
        face = cv2.imread('test_imgs/face.jpg')
        l_eye, r_eye = crop_eyes(face)

        self.assertIsNotNone(l_eye)
        self.assertIsNotNone(r_eye)

    def test_cnn_preprocess(self):
        face = cv2.imread('test_imgs/face.jpg')

        # convert image to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        img = cnn_preprocess(gray)

        self.assertTrue(((0 <= img) & (img <= 1)).all())

    def test_build_model(self):
        model = build_model()

        self.assertIsInstance(model, Sequential)