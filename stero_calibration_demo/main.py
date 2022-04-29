"""
Author: windzu
Date: 2022-04-28 11:51:31
LastEditTime: 2022-04-28 11:51:32
LastEditors: windzu
Description: 
FilePath: /demo_stew/stero_calibration_demo/main.py
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
        files.sort()
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                # temp
                print(os.path.join(root, file))
                image_list.append(cv2.imread(os.path.join(root, file)))
    if len(image_list) == 0:
        raise Exception("No image found in {}".format(image_path))
    image_size = image_list[0].shape[:2]
    return image_list, image_size


def main():
    board_size = (8, 5)  # (cols,rows)
    square_size = 1
    BOARD = np.array(
        [[(j * square_size, i * square_size, 0.0)] for i in range(board_size[1]) for j in range(board_size[0])], dtype=np.float32,
    )

    # 预先标定好的左右相机
    left_intrinsics_matrix = np.array([[4190.74, 0.0, 397.534], [0.0, 5016.19, 552.973], [0.0, 0.0, 1.0],])
    left_distortion_coefficients = np.array([1.26191554, 9.95544452, 0.18353262, -0.52436386, -54.48754767])
    right_intrinsics_matrix = np.array([[1563.49, 0.0, 378.762], [0.0, 1467.98, 626.4630], [0.0, 0.0, 1.0],])
    right_distortion_coefficients = np.array([0.39453957, 0.05506374, 0.0978351, -0.13053234, -0.18893312])

    left_image_path = "./data/left/"
    right_image_path = "./data/right/"

    left_image_list, image_size = read_image_directory(left_image_path)
    right_image_list, image_size = read_image_directory(right_image_path)

    if len(left_image_list) != len(right_image_list):
        raise Exception("Image number not equal")

    left_corners = get_corners(left_image_list, board_size)
    right_corners = get_corners(right_image_list, board_size)

    board_list = [BOARD for i in range(len(left_image_list))]

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    (
        reproj_err,
        left_intrinsics_matrix,
        left_distortion_coefficients,
        right_intrinsics_matrix,
        right_distortion_coefficients,
        R,
        T,
        E,
        F,
    ) = cv2.stereoCalibrate(
        board_list,
        left_corners,
        right_corners,
        left_intrinsics_matrix,
        left_distortion_coefficients,
        right_intrinsics_matrix,
        right_distortion_coefficients,
        image_size,
        criteria=criteria,
        flags=stereocalibration_flags,
    )
    print("reproj_err:", reproj_err)
    print("R:", R)
    print("T:", T)
    print("E:", E)
    print("F:", F)


if __name__ == "__main__":
    main()
