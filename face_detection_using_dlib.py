import cv2
import dlib
import time

if __name__ == '__main__':

    # get frames from webcam
    cap = cv2.VideoCapture(0)

    # get frontal face detector from dlib
    detector = dlib.get_frontal_face_detector()

    # load the cascade for the eyes
    eye_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_eye.xml')

    fps_vector = []
    face_vetor = []

    while True:
        start = time.time()

        # read from the video stream
        ret, frame = cap.read()

        # transform every frame in gray scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:
            x, y = face.left(), face.top()
            w, h = face.right(), face.bottom()

            # paint a rectangle around the face
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 3)

            # get the region of interest (face) in the black and white image
            # get the upper half of the face
            h = (h + y) // 2
            roi_gray = gray[y:h, x:w]

            # get the region of interest (face) in the colored image
            roi_color = frame[y:h, x:w]

            # apply the detectMultiScale method to locate one or several eyes on the face
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

            # for each detected eye:
            for (ex, ey, ew, eh) in eyes:

                # paint a rectangle around the eyes, but inside the referential of the face
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

                eye = roi_gray[ey: ey + eh, ex: ex + ew]
                cv2.imshow('eye', eye)

        cv2.imshow("cropped gray face", roi_gray)
        cv2.imshow("frame", frame)

        # If typed on the keyboard:
        if cv2.waitKey(1) == ord('s'):

            # stop the loop
            break

        end = time.time()
        seconds = end - start
        fps = 5.0 / seconds
        fps_vector.append(fps)
        face_vetor.append(len(faces))
        print('faces: %.2f' % len(faces))
        print('fps: %.2f' % fps)

    average_fps = sum(fps_vector) / len(fps_vector)
    averate_faces = sum(face_vetor) / len(face_vetor)

    print("MEAN FPS DLIB: %.2f" % average_fps)
    print("MEAN DETECTED FACES DLIB: %.2f" % averate_faces)

    # turn the webcam off
    cap.release()

    # destroy all the windows inside which the images were displayed
    cv2.destroyAllWindows()
