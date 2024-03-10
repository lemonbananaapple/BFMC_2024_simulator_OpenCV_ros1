import cv2
import numpy as np




def binary_filter_road_pavement(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Tính toán các tham số
    roihist = cv2.calcHist([img_hsv], [0, 1], None, [256, 256], [0, 256, 0, 256])

    # Chuẩn hóa lược đồ màu và áp dụng phản chiếu ngược
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([img_hsv], [0, 1], roihist, [0, 256, 0, 256], 1)

    # Lọc ảnh với cấu trúc hình dạng elip
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cv2.filter2D(dst, -1, disc, dst)

    # Ngưỡng và áp dụng phép AND nhị phân
    ret, thresh = cv2.threshold(dst, 10, 250, cv2.THRESH_BINARY_INV)

    return thresh

def combined_threshold(img, kernel=3, grad_thresh=(30, 100), mag_thresh=(70, 100), dir_thresh=(0.8, 0.9),
                       s_thresh=(100, 255), r_thresh=(150, 255), u_thresh=(140, 180),
                       ):

    def binary_thresh(channel, thresh=(200, 255), on=1):
        binary = np.zeros_like(channel)
        binary[(channel > thresh[0]) & (channel <= thresh[1])] = on
        return binary


    # Chuyển ảnh sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lấy đạo hàm gradient theo cả x và y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

    # Tính toán đạo hàm gradient theo x
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    gradx = binary_thresh(scaled_sobelx, grad_thresh)

    # Tính toán đạo hàm gradient theo y
    abs_sobely = np.absolute(sobely)
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    grady = binary_thresh(scaled_sobely, grad_thresh)

    # Tính toán độ lớn gradient
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    mag_binary = binary_thresh(gradmag, mag_thresh)

    # Lấy giá trị tuyệt đối của hướng gradient,
    # Áp dụng ngưỡng và tạo ảnh nhị phân kết quả
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = binary_thresh(absgraddir, dir_thresh)

    # Chuyển đổi sang không gian màu HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    sbinary = binary_thresh(S, s_thresh)

    # Lấy giá trị của kênh màu đỏ
    R = img[:, :, 2]
    rbinary = binary_thresh(R, r_thresh)

    # Chuyển đổi sang không gian màu YUV
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    U = yuv[:, :, 1]
    ubinary = binary_thresh(U, u_thresh)

    # Áp dụng ngưỡng cho trường hợp "daytime-bright"
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (sbinary == 1) | (rbinary == 1)] = 1

    return combined



def binary_filter_road_pavement(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create dimensions for roi boxes - center, and left & right half height
    width = img.shape[1]
    height = img.shape[0]
    x_center = np.int(width / 2)
    roi_width = 100
    roi_height = 200
    x_left = x_center - int(roi_width / 2)
    x_right = x_center + int(roi_width / 2)
    y_top = height - 30
    y_bottom = y_top - roi_height
    y_bottom_small = y_top - int(roi_height / 2)
    x_offset = 50
    x_finish = width - x_offset

    # extract the roi and stack before converting to HSV
    roi_center = img[y_bottom:y_top, x_left:x_right]
    roi_left = img[y_bottom_small:y_top, x_offset:roi_width + x_offset]
    roi_right = img[y_bottom_small:y_top, x_finish - roi_width:x_finish]
    roi = np.hstack((roi_center, np.vstack((roi_left, roi_right))))
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # calculating object histogram
    roihist = cv2.calcHist([roi_hsv], [0, 1], None, [256, 256], [0, 256, 0, 256])
    #     roihist = cv2.calcHist([roi_hsv],[0], None, [256], [0, 256] )

    # normalize histogram and apply backprojection
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([img_hsv], [0, 1], roihist, [0, 256, 0, 256], 1)
    #     dst = cv2.calcBackProject([img_hsv],[0],roihist,[0,256],1)

    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cv2.filter2D(dst, -1, disc, dst)

    # threshold and binary AND
    ret, thresh = cv2.threshold(dst, 10, 250, cv2.THRESH_BINARY_INV)

    return thresh


