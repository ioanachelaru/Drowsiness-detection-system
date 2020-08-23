import cv2
import dlib
from imutils import face_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from pygame import mixer

# dlib shape predictor used for the eyes
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# face predictor uses haarcascade file
face_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_alt.xml')

# init the mixer for alarm playing
mixer.init()
sound = mixer.Sound('alarm.wav')


def detect_face(img, cascade=face_cascade, minimum_feature_size=(20, 20)):
    # if the cascade argument is missing
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))

    # detect faces
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimum_feature_size)

    # if it doesn't return rectangle return empty array
    if len(rects) == 0:
        return []

    #  convert last coord from (width, height) to (maxX, maxY)
    rects[:, 2:] += rects[:, :2]

    return rects


def crop_eyes(frame):
    # convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the face at grayscale image
    te = detect_face(gray, minimum_feature_size=(80, 80))

    # if the face detector doesn't detect face
    # return None, else if detects more than one faces
    # keep the bigger and if it is only one keep one dim
    if len(te) == 0:
        return None
    elif len(te) > 1:
        face = te[0]
    elif len(te) == 1:
        [face] = te

    # draw rectangle around the face
    cv2.rectangle(frame, (int(face[0]), int(face[1])), (int(face[2]), int(face[3])), (0, 255, 0), 2)

    # keep the face region from the whole frame
    face_rect = dlib.rectangle(left=int(face[0]), top=int(face[1]),
                               right=int(face[2]), bottom=int(face[3]))

    # determine the facial landmarks for the face region
    shape = predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)

    #  grab the indexes of the facial landmarks for the left and right eye
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # extract the left and right eye coordinates
    left_eye = shape[lStart:lEnd]
    right_eye = shape[rStart:rEnd]

    # keep the upper and the lower limit of the eye
    # and compute the height
    l_uppery = min(left_eye[1:3, 1])
    l_lowy = max(left_eye[4:, 1])
    l_dify = abs(l_uppery - l_lowy)

    # compute the width of the eye
    lw = (left_eye[3][0] - left_eye[0][0])

    # the image for the cnn must be (26,34)
    # add half of the difference at x and y
    # axis from the width at height respectively left-right
    # and up-down
    minxl = (left_eye[0][0] - ((34 - lw) / 2))
    maxxl = (left_eye[3][0] + ((34 - lw) / 2))
    minyl = (l_uppery - ((26 - l_dify) / 2))
    maxyl = (l_lowy + ((26 - l_dify) / 2))

    # crop the eye rectangle from the frame
    left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
    left_eye_rect = left_eye_rect.astype(int)
    left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]

    # same as left eye for right eye
    r_uppery = min(right_eye[1:3, 1])
    r_lowy = max(right_eye[4:, 1])
    r_dify = abs(r_uppery - r_lowy)
    rw = (right_eye[3][0] - right_eye[0][0])
    minxr = (right_eye[0][0] - ((34 - rw) / 2))
    maxxr = (right_eye[3][0] + ((34 - rw) / 2))
    minyr = (r_uppery - ((26 - r_dify) / 2))
    maxyr = (r_lowy + ((26 - r_dify) / 2))
    right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
    right_eye_rect = right_eye_rect.astype(int)
    right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

    # if it doesn't detect left or right eye return None
    if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
        return None

    # resize eyes for the convolutional neural network
    left_eye_image = cv2.resize(left_eye_image, (34, 26))
    right_eye_image = cv2.resize(right_eye_image, (34, 26))

    # flip the right eye image
    right_eye_image = cv2.flip(right_eye_image, 1)

    # return left and right eye
    return left_eye_image, right_eye_image


# process the image to have the same format as the training data
def cnn_preprocess(img):
    img = img.astype('float32')

    # normalize the image
    img /= 255

    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)

    return img


def build_model():
    # dimensions of the photos
    height = 26
    width = 34
    dims = 1

    loss_function = 'binary_crossentropy'
    learning_rate = 0.001

    model = Sequential()

    # first layer of convolution
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(height, width, dims)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second layer of convolution
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # third layer of convolution
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # drops some neurons along with their weights to avoid overfitting
    model.add(Dropout(0.25))

    model.add(Flatten())

    # first dense layer
    model.add(Dense(512))
    model.add(Activation('relu'))

    # second dense layer
    model.add(Dense(512))
    model.add(Activation('relu'))

    # output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_function, metrics=['accuracy'])

    return model


def detect(camera):
    # set parameters for the output
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 0
    thicc = 2

    # load model
    model = build_model()
    model.load_weights('models/eyeClassifier_v6.hdf5')

    while True:

        ret, frame = camera.read()

        height, width = frame.shape[:2]

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        # detect eyes
        eyes = crop_eyes(frame)

        if eyes is None:
            continue
        else:
            left_eye, right_eye = eyes

        # print both eyes
        # cv2.imshow('ochi stang', left_eye)
        # cv2.imshow('ochi drept', right_eye)

        prediction_right_eye = model.predict(cnn_preprocess(right_eye)) > 0.5
        prediction_left_eye = model.predict(cnn_preprocess(left_eye)) > 0.5

        # print eyes predictions
        # TRUE - OPEN
        # FALSE - CLOSE
        print('Right eye', prediction_right_eye, 'Left eye', prediction_left_eye)

        # if both eyes are closed
        if prediction_right_eye == [[False]] and prediction_left_eye == [[False]]:

            # increace the score
            score = score + 1

            # print Closed on the frame
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # at least one eye is open
        else:

            # decreace the score
            score = score - 1

            # print Open on the frame
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # no negative score allowed
        if score < 0:
            score = 0

        # print the score on the frame
        cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # if score gratter than 15
        if score > 15:

            # person is feeling sleepy so we play the alarm
            try:
                sound.play()
            except:
                pass

            # increace thicc when necessary
            if thicc < 16:
                thicc = thicc + 2

            # decreace thicc when necessary
            else:
                thicc = thicc - 2

                # no thicc <=1 allowed
                if thicc < 2:
                    thicc = 2

            # print thicc border on the frame
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        else:
            try:
                sound.stop()
            except:
                pass

        # show the frame
        cv2.imshow('Drowsiness detection', frame)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            return


def main():
    # open the camera
    camera = cv2.VideoCapture(0)

    detect(camera)

    # clean up
    cv2.destroyAllWindows()
    del camera


if __name__ == '__main__':
    main()
