import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks


def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d


def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector,
                             translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]

    return (x, y)


def get_angles(rvec, tvec):
    rmat = cv2.Rodrigues(rvec)[0]
    P = np.hstack((rmat, tvec))  # projection matrix [R | t]
    degrees = -cv2.decomposeProjectionMatrix(P)[6]
    rx, ry, rz = degrees[:, 0]
    return [rx, ry, rz]


face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corne
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

sideways = [0]
left_right = [0]
front_back = [0]
angle_sideways_max= 0
angle_sideways_min= 180
angle_front_back_max=0
angle_front_back_min=180
angle_left_right_max=0
angle_left_right_min=180

while True:
    ret, img = cap.read()
    if ret == True:
        faces = find_faces(img, face_model)
        for face in faces:
            marks = detect_marks(img, landmark_model, face)
            image_points = np.array([
                                    marks[30],     # Nose tip
                                    marks[8],      # Chin
                                    marks[36],     # Left eye left corner
                                    marks[45],     # Right eye right corne
                                    marks[48],     # Left Mouth corner
                                    marks[54]      # Right mouth corner
                                    ], dtype="double")

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)

            (x, jacobian) = cv2.projectPoints(np.array(
                [(500.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            (y, jacobian) = cv2.projectPoints(np.array(
                [(0.0, 500.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            (z, jacobian) = cv2.projectPoints(np.array(
                [(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            org = (int(image_points[0][0]), int(image_points[0][1]))

            x_proj = (int(x[0][0][0]), int(x[0][0][1]))
            cv2.line(img, org, x_proj, (255, 0, 0), 2)

            y_proj = (int(y[0][0][0]), int(y[0][0][1]))
            cv2.line(img, org, y_proj, (0, 255, 0), 2)

            z_proj = (int(z[0][0][0]), int(z[0][0][1]))
            cv2.line(img, org, z_proj, (0, 0, 255), 2)

            angles = get_angles(rotation_vector, translation_vector)

            angle_sideways = round(angles[2])
            angle_left_right = round(angles[1])
            angle_front_back = round(angles[0])-180
            if(angle_front_back < -180):
                angle_front_back += 360
            
            if(angle_sideways_max < angle_sideways):
                angle_sideways_max = angle_sideways
                
            if(angle_sideways_min > angle_sideways):
                angle_sideways_min = angle_sideways
                
            if(angle_front_back_max < angle_front_back):
                angle_front_back_max = angle_front_back
                
            if(angle_front_back_min > angle_front_back):
                angle_front_back_min = angle_front_back
                
            if(angle_left_right_max < angle_left_right):
                angle_left_right_max = angle_left_right
                
            if(angle_left_right_min > angle_left_right):
                angle_left_right_min = angle_left_right
                

            cv2.putText(img, 'sideways: ' + str(angle_sideways),
                        (30, 30), font, 1, (255, 0, 0), 2)
            cv2.putText(img, 'Max sideways: ' + str(angle_sideways_max),
                        (30, 60), font, 1, (255, 100, 0), 2)
            cv2.putText(img, 'Min sideways: ' + str(angle_sideways_min),
                        (30, 90), font, 1, (255, 0, 100), 2)
            cv2.putText(img, 'left-right: ' + str(angle_left_right),
                        (30, 120), font, 1, (0, 0, 255), 2)
            cv2.putText(img, 'Max left-right: ' + str(angle_left_right_max),
                        (30, 150), font, 1, (100, 0, 255), 2)
            cv2.putText(img, 'Min left-right: ' + str(angle_left_right_min),
                        (30, 180), font, 1, (0, 100, 255), 2)
            cv2.putText(img, 'front-back: ' + str(angle_front_back),
                        (30, 210), font, 1, (0, 255, 0), 2)
            cv2.putText(img, 'Max front-back: ' + str(angle_front_back_max),
                        (30, 240), font, 1, (100, 255, 0), 2)
            cv2.putText(img, 'Min front-back: ' + str(angle_front_back_min),
                        (30, 270), font, 1, (0, 255, 100), 2)

            if abs(angle_sideways-sideways[-1])<=50:
                sideways.append(angle_sideways)
            
            if abs(angle_left_right-left_right[-1])<=50:
                left_right.append(angle_left_right)
                
            if abs(angle_front_back-front_back[-1])<=50:
                front_back.append(angle_front_back)

        cv2.imshow('Head Pose Estimation', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()

sideways.sort()
left_right.sort()
front_back.sort()

print("Sideways:", sideways[0], sideways[-1])
print("Left Right:", left_right[0], left_right[-1])
print("Front Back:", front_back[0], front_back[-1])