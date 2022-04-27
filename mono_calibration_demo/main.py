"""
Author: windzu
Date: 2022-04-27 10:49:04
LastEditTime: 2022-04-27 15:39:17
LastEditors: windzu
Description: 
FilePath: /demo_stew/mono_calibration_demo/main.py
@Copyright (C) 2021-2022 xxx Company Limited. All rights reserved.
@Licensed under the Apache License, Version 2.0 (the License)
"""
import os
import cv2
import numpy as np


def get_corners(image_list, board_size):
    """
    Get corners of chessboard
    """
    corners_list = []
    for image in image_list:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
            )
            corners_list.append(corners)
            cv2.drawChessboardCorners(image, board_size, corners, ret)
            cv2.imshow("image", image)
            cv2.waitKey(100)

    return corners_list


def read_image_directory(image_path):
    image_list = []
    for root, dirs, files in os.walk(image_path):
        for file in files:
            if file.endswith(".jpg"):
                image_list.append(cv2.imread(os.path.join(root, file)))
    if len(image_list) == 0:
        raise Exception("No image found in {}".format(image_path))
    image_size = image_list[0].shape[:2]
    return image_list, image_size


def pinhole_calibration(image_path, board_size, BOARD):
    """
    Pinhole camera calibration
    """
    image_path = image_path + "pinhole/"
    image_list, image_size = read_image_directory(image_path)
    board_list = [BOARD for i in range(len(image_list))]

    corners_list = get_corners(image_list, board_size)
    reproj_err, intrinsics_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
        board_list, corners_list, image_size, np.eye(3, 3), np.zeros((5, 1)), flags=0
    )
    print("Reprojection error:", reproj_err)


def fisheye_calibration(image_path, board_size, BOARD):
    """
    Fisheye camera calibration
    """
    image_path = image_path + "fisheye/"
    image_list, image_size = read_image_directory(image_path)
    board_list = [BOARD for i in range(len(image_list))]

    corners_list = get_corners(image_list, board_size)
    reproj_err, intrinsics_matrix, distortion_coefficients, rvecs, tvecs = cv2.fisheye.calibrate(
        board_list,
        corners_list,
        image_size,
        np.eye(3, 3),
        np.zeros((4, 1)),
        flags=cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    )
    print("Reprojection error:", reproj_err)


def main():
    image_path = "./data/"
    board_size = (8, 5)  # (cols,rows)
    square_size = 0.025
    BOARD = np.array(
        [[(j * square_size, i * square_size, 0.0)] for i in range(board_size[1]) for j in range(board_size[0])], dtype=np.float32,
    )
    # 1. Pinhole camera calibration
    # pinhole_calibration(image_path=image_path, board_size=board_size,  BOARD=BOARD)
    # 2. Fisheye camera calibration
    fisheye_calibration(image_path=image_path, board_size=board_size, BOARD=BOARD)


if __name__ == "__main__":
    main()
