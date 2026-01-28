#!/usr/bin/env python
# coding: utf-8

# In[2]:


!pip install mediapipe opencv-python

import mediapipe as mp
import cv2

print("MediaPipe başarıyla yüklendi!")
print("OpenCV versiyon:", cv2.__version__)

# In[1]:


import cv2
import mediapipe as mp
import numpy as np

# MediaPipe modülleri
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Iris ve göz landmark indeksleri
LEFT_EYE_LANDMARKS = [33, 133]   # sol göz yatay uç noktalar
RIGHT_EYE_LANDMARKS = [362, 263] # sağ göz yatay uç noktalar
LEFT_IRIS = [468, 469, 470, 471] # sol iris noktaları
RIGHT_IRIS = [473, 474, 475, 476] # sağ iris noktaları

# Kamera aç
cap = cv2.VideoCapture(0)

# FaceMesh başlat
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,     # İris takip için şart!!
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV BGR -> RGB dönüşümü
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            for face in result.multi_face_landmarks:
                
                #1. YÜZ ÇİZİMİ
                mp_drawing.draw_landmarks(
                    frame, face, mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )

                h, w, _ = frame.shape

                # Landmark koordinatlarını al
                mesh_points = np.array(
                    [(int(p.x * w), int(p.y * h)) for p in face.landmark]
                )

                #2. GÖZ TAKİBİ (sol ve sağ göz uç noktaları)
                left_eye = mesh_points[LEFT_EYE_LANDMARKS]
                right_eye = mesh_points[RIGHT_EYE_LANDMARKS]

                cv2.circle(frame, tuple(left_eye[0]), 3, (0,255,255), -1)
                cv2.circle(frame, tuple(left_eye[1]), 3, (0,255,255), -1)
                cv2.circle(frame, tuple(right_eye[0]), 3, (255,255,0), -1)
                cv2.circle(frame, tuple(right_eye[1]), 3, (255,255,0), -1)

                #3. İRIS TAKİBİ (iris center)
                left_iris_points = mesh_points[LEFT_IRIS]
                right_iris_points = mesh_points[RIGHT_IRIS]

                # Sol iris merkezi
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = left_iris_points
                left_center = np.mean(left_iris_points, axis=0).astype(int)

                # Sağ iris merkezi
                right_center = np.mean(right_iris_points, axis=0).astype(int)

                cv2.circle(frame, tuple(left_center), 4, (0,0,255), -1)
                cv2.circle(frame, tuple(right_center), 4, (0,0,255), -1)

                cv2.putText(frame, "Left Iris", (left_center[0], left_center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                cv2.putText(frame, "Right Iris", (right_center[0], right_center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

        cv2.imshow("Face + Eye + Iris Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
