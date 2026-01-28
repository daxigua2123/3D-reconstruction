import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter

def gaussian_derivative_kernels(sigma=1.0):
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    half = ksize // 2
    x = np.arange(-half, half + 1)
    gauss = np.exp(-x**2 / (2 * sigma**2))
    gauss /= np.sum(gauss)
    gauss1 = -x * np.exp(-x**2 / (2 * sigma**2)) / (sigma**2)
    gauss1 -= np.mean(gauss1)
    gauss2 = (x**2 - sigma**2) * np.exp(-x**2 / (2 * sigma**2)) / (sigma**4)
    gauss2 -= np.mean(gauss2)
    return gauss, gauss1, gauss2

def stegercomputeDerivative(img, sigma=1.0):
    g, g1, g2 = gaussian_derivative_kernels(sigma)
    g = g.reshape(-1, 1)
    g1 = g1.reshape(-1, 1)
    g2 = g2.reshape(-1, 1)

    dx  = cv2.filter2D(img, -1, g1 @ g.T)
    dy  = cv2.filter2D(img, -1, g @ g1.T)
    dxx = cv2.filter2D(img, -1, g2 @ g.T)
    dyy = cv2.filter2D(img, -1, g @ g2.T)
    dxy = cv2.filter2D(img, -1, g1 @ g1.T)

    return dx, dy, dxx, dyy, dxy

def getstegerpoint(dx, dy, dxx, dyy, dxy, img, thresh=1.0, intensity_thresh=50):
    h, w = img.shape
    points = []

    for i in range(h):
        for j in range(w):
            if img[i, j] < intensity_thresh:
                continue
            H = np.array([[dxx[i,j], dxy[i,j]],
                          [dxy[i,j], dyy[i,j]]], dtype=float)
            ret, eigenVal, eigenVect = cv2.eigen(H)
            if not ret:
                continue
            if abs(eigenVal[0, 0]) > abs(eigenVal[1, 0]):
                lam = eigenVal[0, 0]
                nx, ny = eigenVect[0, 0], eigenVect[0, 1]
            else:
                lam = eigenVal[1, 0]
                nx, ny = eigenVect[1, 0], eigenVect[1, 1]

            if abs(lam) < thresh:
                continue

            denom = dxx[i,j]*nx*nx + dyy[i,j]*ny*ny + 2*dxy[i,j]*nx*ny
            if denom == 0:
                continue

            T = -(dx[i,j]*nx + dy[i,j]*ny) / denom

            if abs(T*nx) <= 0.5 and abs(T*ny) <= 0.5:
                sub_y = i + T * ny
                sub_x = j + T * nx
                points.append((sub_y, sub_x, img[i, j]))

    return points

def nms_column(points, neighborhood=10):
    col_dict = {}
    for y, x, inten in points:
        xi = int(round(x))
        col_dict.setdefault(xi, []).append((y, inten))

    output = []
    for x, arr in col_dict.items():
        arr.sort(key=lambda t: t[1], reverse=True)  
        keep = []
        for y, inten in arr:
            if all(abs(y - yy) > neighborhood for yy, _ in keep):
                keep.append((y, inten))
        for y, inten in keep:
            output.append((y, x, inten))
    return output

def akima_interpolation(nms_points, img, smooth=True):
    x_dict = {}
    for y, x, inten in nms_points:
        xi = int(round(x))
        if xi not in x_dict or inten > x_dict[xi][1]:
            x_dict[xi] = (y, inten)
    x_sorted = sorted(x_dict.keys())
    y_sorted = [x_dict[x][0] for x in x_sorted]
    if len(x_sorted) < 2:
        return []
    akima = Akima1DInterpolator(x_sorted, y_sorted)
    full_x = np.arange(min(x_sorted), max(x_sorted)+1)
    full_y = akima(full_x)
    valid_mask = ~np.isnan(full_y)
    full_x, full_y = full_x[valid_mask], full_y[valid_mask]
    valid_idx = (full_y >= 0) & (full_y < img.shape[0])
    full_x, full_y = full_x[valid_idx], full_y[valid_idx]
    if smooth and len(full_y) > 7:
        full_y = savgol_filter(full_y, window_length=7, polyorder=3)
    filtered_points = []
    for y, x in zip(full_y, full_x):
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < img.shape[0] and 0 <= ix < img.shape[1]:
            if img[iy, ix] > 0:
                filtered_points.append((y, x))
    return filtered_points

def getcenterline(img, visualization=False):
    npimg = img.astype(float)
    blur_img = cv2.GaussianBlur(npimg, (9, 9), 1.0)
    
    DX, DY, DXX, DYY, DXY = stegercomputeDerivative(blur_img)
   
    steger_pts = getstegerpoint(DX, DY, DXX, DYY, DXY,
                                blur_img, thresh=1.0, intensity_thresh=50)
   
    nms_pts = nms_column(steger_pts, neighborhood=5)
    
    interp_pts = akima_interpolation(nms_pts, img, smooth=True)

    # ---------- visualization ----------
    if visualization:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        plt.figure(figsize=(8,6))
        plt.imshow(img_rgb)
        for y, x, _ in steger_pts:
            plt.plot(x, y, 'g.', markersize=2)
        for y, x in interp_pts:
            plt.plot(x, y, 'r.', markersize=3)

        plt.title("Centerline Extraction")
        plt.show()

    return interp_pts
