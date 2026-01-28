# -*- coding: utf-8 -*-
from calibration import Calibrator as Calibrator
import numpy as np
def main():
    img_dir = "......."   
    calibrator = Calibrator(img_dir)
    calibrator.visualization()
    # calibrate the camera
    mat_intri, coff_dis,r,t = calibrator.calibrate_camera()
    print(mat_intri)
    print(coff_dis)
    # calibrator.visualization()
    np.savez('camera_para_260113_3.npz', intri=mat_intri, dis=coff_dis)
if __name__ == '__main__':
    main()
