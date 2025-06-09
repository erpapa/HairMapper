"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect and crop the human faces in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

For more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""

import cv2
import numpy as np
from argparse import ArgumentParser
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils.utils import refine

def rotation_matrix_to_euler_angles(rmat):
    '''
    Ref: https://stackoverflow.com/a/15029416
    '''
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)

    if sy < 1e-6:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0
    else:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])

    return np.degrees([x, y, z])

def run_head_pose():
    # Get the frame size. This will be used by the following detectors.
    frame_width = 512
    frame_height = 512

    # Setup a face detector to detect human faces.
    face_detector = FaceDetector('../ckpts/head_pose_model/face_detector.onnx')

    # Setup a mark detector to detect landmarks.
    mark_detector = MarkDetector('../ckpts/head_pose_model/face_landmarks.onnx')

    # Setup a pose estimator to solve pose.
    pose_estimator = PoseEstimator(frame_width, frame_height)

    img = cv2.imread('../test_data/image/001.png')
    img = cv2.resize(img, [frame_width,frame_height])
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Step 1: Get faces from current img.
    faces, _ = face_detector.detect(frame, 0.7)
    # Any valid face found?
    if len(faces) > 0:
        # Step 2: Detect landmarks. Crop and feed the face area into the
        # mark detector. Note only the first face will be used for
        # demonstration.
        face = refine(faces, frame_width, frame_height, 0.15)[0]
        x1, y1, x2, y2 = face[:4].astype(int)
        patch = frame[y1:y2, x1:x2]

        # Run the mark detection.
        marks = mark_detector.detect([patch])[0].reshape([68, 2])

        # Convert the locations from local face area to the global image.
        marks *= (x2 - x1)
        marks[:, 0] += x1
        marks[:, 1] += y1

        # Step 3: Try pose estimation with 68 points.
        pose = pose_estimator.solve(marks)

        rotation_vector, translation_vector = pose
        mdists = pose_estimator.dist_coeefs
        camera_matrix = pose_estimator.camera_matrix
        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotation_vector, translation_vector, camera_matrix, mdists)

        # calculating angle
        rmat, jac = cv2.Rodrigues(rotation_vector)
        euler_angles = rotation_matrix_to_euler_angles(rmat)
        print('euler_angles: ', euler_angles)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
         
        print('*' * 80)
        print("Angle: ", angles)
        # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
        x = np.arctan2(Qx[2][1], Qx[2][2])
        y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
        z = np.arctan2(Qz[0][0], Qz[1][0])
        print("AxisX: ", x)
        print("AxisY: ", y)
        print("AxisZ: ", z)
        print('*' * 80)

        # All done. The best way to show the result would be drawing the
        # pose on the frame in realtime.

        # Do you want to see the pose annotation?
        # pose_estimator.visualize(frame, pose, color=(0, 255, 0))

        # Do you want to see the axes?
        # pose_estimator.draw_axes(frame, pose)

        # Do you want to see the marks?
        # mark_detector.visualize(frame, marks, color=(0, 255, 0))

        # Do you want to see the face bounding boxes?
        # face_detector.visualize(frame, faces)


if __name__ == '__main__':
    run_head_pose()
