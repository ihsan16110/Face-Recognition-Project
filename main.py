import face_recognition
from PIL import Image, ImageDraw
import numpy as np

# Load a sample picture and learn how to recognize it.

import cv2

video_capture =cv2.VideoCapture(0)
mahdi_image = face_recognition.load_image_file("mahdi.jpg")
mahdi_image_face_encoding = face_recognition.face_encodings(mahdi_image)[0]

# Load a second sample picture and learn how to recognize it.
alamin_image = face_recognition.load_image_file("Alamin.jpg")
alamin_image_face_encoding = face_recognition.face_encodings(alamin_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    mahdi_image_face_encoding,
    alamin_image_face_encoding
]
known_face_names = [
    "Mahdi",
    "Alamin"
]


while True:
     ret, frame = video_capture.read()
     rgb_frame = frame[:, :, :: -1]

     face_loacations = face_recognition.face_locations(rgb_frame)
     face_encodings = face_recognition.face_encodings(rgb_frame, face_loacations)
     for (top, right, bottom, left), face_encoding in zip(face_loacations, face_encodings):
          matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

          name = "Unknown"

          if True in matches:
               first_match_index = matches.index(True)
               name = known_face_names[first_match_index]

          #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
          cv2.rectangle(frame,(left, bottom - 35),(right,bottom),(0,0,255), cv2.FILLED)
          font= cv2.FONT_HERSHEY_DUPLEX
          cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)

     cv2.imshow('Video', frame)

     if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindow()
# pil_image.save("image_with_boxes.jpg")