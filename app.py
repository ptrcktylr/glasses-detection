from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import cv2
import numpy as np
import imutils
import time
import os


def detect_glasses(frame, faceNet, glassesNet):
    # make blob from image
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # give blob to network and get face detections
    faceNet.setInput(blob)
    face_detections = faceNet.forward()

    # init list of faces and their locations along
    # with list of predictions from glasses network
    faces = []
    locations = []
    predictions = []

    # for each detection
    for i in range(0, face_detections.shape[2]):
        # get confidence for each detection
        confidence = face_detections[0, 0, i, 2]

        # filter out low confidence detections
        if confidence > 0.5:
            # create the box around face
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # make sure the boxes are in frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # get the face region of interest
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add face and box to lists
            faces.append(face)
            locations.append((startX, startY, endX, endY))

    # only make glasses predictions if we detect faces
    if len(faces) > 0:
        faces = np.array(faces, dtype='float32')
        predictions = glassesNet.predict(faces, batch_size=32)

    # return tuple with locations & predictions
    return (locations, predictions)


# load face detection model from https://github.com/opencv/opencv/tree/3.4.0/samples/dnn
prototxt = r'face_detection/deploy.prototxt'
caffemodel = r'face_detection/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxt, caffemodel)

# load the glasses detection model we made in train.py
glassesNet = load_model('glasses_detection.model')

# start the video stream from webcam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over video frames
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    # detect faces in frame and determine if they're
    # wearing glasses or not
    (locations, predictions) = detect_glasses(frame, faceNet, glassesNet)

    # loop over detected locations
    for (box, prediction) in zip(locations, predictions):
        (startX, startY, endX, endY) = box
        (glasses, withoutGlasses) = prediction

        # determine label
        label = "Glasses" if glasses > withoutGlasses else "No Glasses"
        color = (0, 255, 0) if label == "Glasses" else (0, 0, 255)

        # label text with probability
        label = "{}: {:.2f}%".format(label, max(glasses, withoutGlasses) * 100)

        # display box & label on frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # quit using q key
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
