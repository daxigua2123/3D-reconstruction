import os
import glob
import cv2
import numpy as np


class Calibrator(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.mat_intri = None
        self.coff_dis = None

      
        self.chessboard_size = (19, 15)   
        self.square_size = 2.5          

        # --------------------------
        # 构建棋盘格世界坐标
        # --------------------------
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                               0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        self.cp_world = []
        self.cp_img = []

        
        self.img_paths = []
        for ext in ["jpg", "png", "jpeg", "tif", "bmp"]:
            self.img_paths += glob.glob(os.path.join(img_dir, f"*.{ext}"))

        assert len(self.img_paths), "No images found!"

        for img_path in self.img_paths:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                self.cp_world.append(objp)
                self.cp_img.append(corners.reshape(-1, 2))

    def visualization(self):
        radius = 5
        color = (0, 0, 255)
        thickness = -1

        for idx, img_path in enumerate(self.img_paths):
            img = cv2.imread(img_path)
            if idx >= len(self.cp_img):
                continue

            for p in self.cp_img[idx]:
                pt = (int(p[0]), int(p[1]))
                cv2.circle(img, pt, radius, color, thickness)

            cv2.namedWindow("corners", 0)
            cv2.imshow("corners", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    def calibrate_camera(self):

        ret, mat_intri, coff_dis, v_rot, v_trans = cv2.calibrateCamera(
            self.cp_world,
            self.cp_img,
            (2048, 2448),
            None,
            None
        )

        total_error = 0
        for i in range(len(self.cp_world)):
            reproject_pts, _ = cv2.projectPoints(
                self.cp_world[i], v_rot[i], v_trans[i], mat_intri, coff_dis
            )
            error = cv2.norm(self.cp_img[i], reproject_pts.reshape(-1, 2), cv2.NORM_L2) / len(reproject_pts)
            total_error += error

        print("Average reprojection error:", total_error / len(self.cp_world))

        return mat_intri, coff_dis, v_rot, v_trans
