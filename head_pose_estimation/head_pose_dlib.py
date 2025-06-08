"""Demo code showing how to estimate human head pose.

For more details, please refer to:
https://github.com/by-sabbir/HeadPoseEstimation
"""

import cv2
import dlib
import numpy as np
import reference_world as world

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
    focal = 1
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../ckpts/shape_predictor_68_face_landmarks.dat')
    
    img = cv2.imread('../test_data/image/001.png')
    img = cv2.resize(img, [512,512])
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(frame, 0)
    face3Dmodel = world.ref3DModel()
    for face in faces:
        shape = predictor(frame, face)
        refImgPts = world.ref2dImagePoints(shape)
        height, width, channel = frame.shape
        focalLength = focal * width
        cameraMatrix = world.cameraMatrix(focalLength, (width / 2, height / 2))
        mdists = np.zeros((4, 1), dtype=np.float64)
        # calculate rotation and translation vector using solvePnP
        success, rotationVector, translationVector = cv2.solvePnP(face3Dmodel, refImgPts, cameraMatrix, mdists)
        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

        # calculating angle
        rmat, jac = cv2.Rodrigues(rotationVector)
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


if __name__ == "__main__":
    run_head_pose()

