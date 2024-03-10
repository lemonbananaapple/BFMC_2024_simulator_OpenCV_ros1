import cv2
import numpy as np
import pickle
from scipy import signal



def lane_histogram(img, height_start, height_end):
    histogram = np.sum(img[int(height_start):int(height_end),:], axis=0)
    return histogram

#sai
def calc_fit_from_boxes(boxes,old_fit):
    if len(boxes) > 0:
        # flaten and adjust all boxes for the binary images
        xs = np.concatenate([b.nonzerox + b.x_left for b in boxes])
        ys = np.concatenate([b.nonzeroy + b.y_bottom for b in boxes])

        # return the polynominal
        return np.polyfit(ys, xs, 2)
    else:
        return old_fit





def find_lane_windows(window_box, binimg):
    boxes = []
    continue_lane_search = True
    contiguous_box_no_line_count = 0

    while continue_lane_search and window_box.y_top >= 0:
        if window_box.has_line():
            # Kiểm tra và điều chỉnh nếu cửa vượt ra khỏi khung hình
            if window_box.x_left < 0:
                window_box.x_left = 0
                window_box.x_center = window_box.x_left + int(window_box.width/2)
            elif window_box.x_right >= binimg.shape[1]:
                window_box.x_right = binimg.shape[1] - 1
                window_box.x_center = window_box.x_right - int(window_box.width/2)

            boxes.append(window_box)

        window_box = window_box.next_windowbox(binimg)

        if window_box.has_lane():
            if window_box.has_line():
                contiguous_box_no_line_count = 0
            else:
                contiguous_box_no_line_count += 1

                if contiguous_box_no_line_count >= 4:
                    continue_lane_search = False

    return boxes

def lane_peaks(histogram):
    peaks = signal.find_peaks_cwt(histogram, np.arange(1, 150), min_length=50)

    midpoint = int(histogram.shape[0] / 2)
    # if we found at least two peaks use the signal approach (better in shadows)
    if len(peaks) > 1:
        # in case more then 2 found just get the left and right one
        peak_left, *_, peak_right = peaks

    # otherwise just choose the highest points in left and right of center segments
    else:

        peak_left = np.argmax(histogram[:midpoint])
        peak_right = np.argmax(histogram[midpoint:]) + midpoint


    return peak_left, peak_right

def camera_calibrate(objpoints, imgpoints, img):
    """Calibrate camera and undistort an image."""
    # Test undistortion on an image
    img_size = img.shape[0:2]
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return ret, mtx, dist, dst


#with open('/home/lucis/PycharmProject/Bosch_car/Lane_detection/calibration_data.pkl', 'rb') as f:
    objpoints, imgpoints, chessboards = pickle.load(f)

# Đọc ảnh để hiệu chỉnh camera


def undistort_image(img, mtx, dist):
    # Assuming mtx and dist are the camera calibration matrices
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def perspective_warp(img, M):
    # img_size = (img.shape[1], img.shape[0])
    img_size = (img.shape[0], img.shape[1])

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped
def perspective_transforms(src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def perspective_unwarp(img, Minv):
    img_size = (img.shape[0], img.shape[1])

    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)

    return unwarped


def calc_warp_points(img_height, img_width, x_center_adj=20):
    imshape = (img_height, img_width)
    xcenter = imshape[1] / 2 + x_center_adj
    xoffset = 0
    xfd = 140
    yf = 300
    src = np.float32([
        (xoffset, imshape[0]),
        (xcenter - xfd, yf),
        (xcenter + xfd, yf),
        (imshape[1] - xoffset, imshape[0])
    ])

    dst = np.float32([
        (xoffset, imshape[1]),
        (xoffset, 0),
        (imshape[0] - xoffset, 0),
        (imshape[0] - xoffset, imshape[1])
    ])

    return src, dst

def poly_fitx(fity, line_fit):
    fit_linex = line_fit[0]*fity**2 + line_fit[1]*fity + line_fit[2]
    return fit_linex

def calc_curvature(poly, height=140):
    fity = np.linspace(0, height - 1, num=height)
    y_eval = np.max(fity)

    # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 13.5/1280 # meters per pixel in y dimension
    # lane_px_height=130 # manual observation
    lane_px_height = 275  # manual observation
    ym_per_pix = (3. / lane_px_height)  # meters per pixel in y dimension
    # xm_per_pix = 6.5/720 # meters per pixel in x dimension
    lane_px_width = 413
    xm_per_pix = 3.7 / lane_px_width

    def fit_in_m(poly):
        xs = poly_fitx(fity, poly)
        xs = xs[::-1]  # Reverse to match top-to-bottom in y

        return np.polyfit(fity * ym_per_pix, xs * xm_per_pix, 2)

    if poly is None:
        return .0

    poly_cr = fit_in_m(poly)
    curveradm = ((1 + (2 * poly_cr[0] * y_eval * ym_per_pix + poly_cr[1]) ** 2) ** 1.5) / np.absolute(2 * poly_cr[0])

    return curveradm


def fit_window(binimg, poly, margin=60):
    height = binimg.shape[0]
    y = binimg.shape[0]

    nonzero = binimg.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    fity = np.linspace(0, height - 1, height)

    def window_lane(poly):
        return (
                (nonzerox > (poly[0] * (nonzeroy ** 2) + poly[1] * nonzeroy + poly[2] - margin))
                & (nonzerox < (poly[0] * (nonzeroy ** 2) + poly[1] * nonzeroy + poly[2] + margin))
        )

    def fit(lane_inds):
        xs = nonzerox[lane_inds]

        return np.polyfit(fity, xs, 2)

    return fit(window_lane(poly))
def calc_lr_fit_from_polys(binimg, left_fit, right_fit, margin):
    nonzero = binimg.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    def window_lane(poly):
        return (
                (nonzerox > (poly[0] * (nonzeroy ** 2) + poly[1] * nonzeroy + poly[2] - margin))
                & (nonzerox < (poly[0] * (nonzeroy ** 2) + poly[1] * nonzeroy + poly[2] + margin))
        )

    def window_polyfit(lane_inds,state_xs,state_ys):
        xs = nonzerox[lane_inds]
        if len(xs) ==0:
            xs=state_xs
        ys = nonzeroy[lane_inds]
        if len(ys) ==0:
            ys=state_ys
        # return the polynominal
        return np.polyfit(ys, xs, 2)


    new_right_fit = right_fit
    if right_fit is not None:
        ys = np.random.uniform(0, 360, 2000)
        xs = np.linspace(230, 240, 2000)

        new_right_fit = window_polyfit(window_lane(right_fit),xs,ys)

    new_left_fit = left_fit
    if left_fit is not None:
        # Nội suy lane nếu left_fit không None
        ys = np.random.uniform(0, 360, 7000)
        xs = np.linspace(0, 10, 7000)
        new_left_fit = window_polyfit(window_lane(left_fit),xs,ys)



    # Assuming 'frame' is your image frame
    frame_test = np.zeros((360, 640, 3), dtype=np.uint8)  # Creating a blank frame for demonstration
    #left_y_values=np.linspace(150,160,1)
    # Generate y values for the left lane
    left_y_values = np.linspace(0, frame_test.shape[0] - 1, frame_test.shape[0])
    left_x_values = np.polyval(new_left_fit, left_y_values)
    #left_x_values=np.mean(left_x_values)
    #right_y_values=np.linspace(150,160,1)

    # Generate y values for the right lane
    right_y_values = np.linspace(0, frame_test.shape[0] - 1, frame_test.shape[0])
    right_x_values = np.polyval(new_right_fit, right_y_values)
    #right_x_values=np.mean(right_x_values)
    # Convert x and y values to integer for drawing
    left_points = np.column_stack((left_x_values.astype(int), left_y_values.astype(int)))

    right_points = np.column_stack((right_x_values.astype(int), right_y_values.astype(int)))
    # Draw circles at each point
    for point in left_points:
        cv2.circle(frame_test, tuple(point), 3, (255, 255, 0), -1)  # -1 indicates filled circle

    for point in right_points:
        cv2.circle(frame_test, tuple(point), 3, (0, 255, 0), -1)  # -1 indicates filled circle

    # Show or save the frame as needed
    cv2.imshow('Lane Detection_test', frame_test)

    return (new_left_fit, new_right_fit)

