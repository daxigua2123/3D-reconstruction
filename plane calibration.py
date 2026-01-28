import numpy as np
from steger import getcenterline
import os
import glob
from calibration import Calibrator as Calibrator
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
def getplanepoint(img_path):

    img_dir = img_path
    # shape_inner_corner = (11, 8)
    # size_grid = 0.02
    # create calibrator
    calibrator = Calibrator(img_dir)
    # calibrate the camera
    mat_intri, coff_dis,r,t = calibrator.calibrate_camera()
    print(mat_intri)
    print(coff_dis)
    cx = mat_intri[0][2]
    cy = mat_intri[1][2]
    fx = mat_intri[0][0]
    fy = mat_intri[1][1]

    allpoint = np.empty((0,3))
    laserimg_dir = "......"
    laserimg_paths = []
    for extension in ["jpg", "png", "jpeg", "tif", "bmp"]:
        laserimg_paths += glob.glob(os.path.join(laserimg_dir, "*.{}".format(extension)))

    for i, img_path in enumerate(laserimg_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        distoredcenterlinelist = getcenterline(img, False)
        nplist = np.array(distoredcenterlinelist)
        distoredcenterline = np.column_stack((nplist[:, 1], nplist[:, 0]))
        centerline = cv2.undistortImagePoints(distoredcenterline,mat_intri, coff_dis).squeeze(axis=1)

        print("第", i + 1, "幅图中心线提取完成")
        laserline = centerline[len(centerline) // 2 - 100:len(centerline) // 2 + 100]
        laserline = np.array(laserline)
        u = laserline[:, 0]
        v = laserline[:, 1]
        xcc = u / fx - cx / fx
        ycc = v / fy - cy / fy
        
        R, _ = cv2.Rodrigues(r[i])
        T = t[i].reshape(3,1)
        s = [0, 0, 1, 0]
        RT = np.hstack((R, T))
        M = np.vstack((RT, np.array([0, 0, 0, 1])))
        patteninccs = s @ np.linalg.inv(M)
        a = patteninccs[0]
        b = patteninccs[1]
        c = patteninccs[2]
        d = patteninccs[3]
        
        denominator = a * xcc + b * ycc + c
        xc = -d * xcc / denominator
        xc = xc.reshape(len(xc),-1)
        yc = (-d * ycc / denominator)
        yc = yc.reshape(len(yc), -1)
        zc = (-d / denominator)
        zc = zc.reshape(len(zc), -1)
        
        point_buff = np.hstack((xc, yc, zc))  
        allpoint = np.vstack((allpoint, point_buff))
    return allpoint
def fitplane(points,visualization = False):  
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    A = np.c_[X, Y, np.ones(X.shape)] 
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None) 
    a, b, c = C
    print(f"拟合平面方程: z = {a:.4f}x + {b:.4f}y + {c:.4f}")

    A_coeff = a
    B_coeff = b
    C_coeff = -1 
    D_coeff = c

    print(f"平面方程的一般形式: {A_coeff:.4f}x + {B_coeff:.4f}y + {C_coeff:.4f}z + {D_coeff:.4f} = 0")

    Z_fit = a * X + b * Y + c
    errors = np.abs(Z - Z_fit)
    mean_error = np.mean(errors)
    print(f"拟合平均误差: {mean_error:.4f}")

    if visualization:
        rcParams['font.sans-serif'] = ['SimHei']  
        rcParams['axes.unicode_minus'] = False 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, color='b', label='原始点')

        x_range = np.linspace(min(X), max(X), 10)
        y_range = np.linspace(min(Y), max(Y), 10)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        z_grid = a * x_grid + b * y_grid + c

        ax.plot_surface(x_grid, y_grid, z_grid, color='r', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title('三维点和拟合平面')

        plt.legend(['原始点'], loc='upper right')
        plt.show()

    return np.array([A_coeff,B_coeff,C_coeff,D_coeff])
def main():
    img_path = "......" 
    point = getplanepoint(img_path)
    plane_PARA = fitplane(point,True)
    np.savez('plane_para_260113_3.npz', plane=plane_PARA)
if __name__ == '__main__':
    main()
