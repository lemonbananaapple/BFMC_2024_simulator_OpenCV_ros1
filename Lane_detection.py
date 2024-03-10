import cv2
import numpy as np
from collections import deque
from define_function import lane_histogram,poly_fitx,calc_fit_from_boxes,find_lane_windows,lane_peaks,calc_curvature,calc_lr_fit_from_polys,perspective_unwarp,calc_warp_points,perspective_transforms,perspective_warp,undistort_image
from preprocessing_img import combined_threshold


class WindowBox(object):
    def __init__(self, binimg, x_center, y_top, width=100, height=40, mincount=100, lane_found=False):
        self.x_center = x_center
        self.y_top = y_top
        self.width = width
        self.height = height
        self.mincount = mincount
        self.lane_found = lane_found
        self.x_left = self.x_center - int(self.width/2)
        self.x_right = self.x_center + int(self.width/2)
        self.y_bottom = self.y_top - self.height
        self.imgwindow = binimg[self.y_bottom:self.y_top, self.x_left:self.x_right]
        self.nonzeroy = self.imgwindow.nonzero()[0]
        self.nonzerox = self.imgwindow.nonzero()[1]

    def center(self):
        return (self.x_center, int(self.y_top - self.y_bottom) / 2)

    def next_windowbox(self, binimg):
        if self.has_line():
            x_center = int(np.mean(self.nonzerox + self.x_left))
        else:
            x_center = self.x_center

        y_top = self.y_bottom

        # Kiểm tra và điều chỉnh nếu cửa vượt ra khỏi khung hình
        if x_center - int(self.width/2) < 0:
            x_center = int(self.width/2)
        elif x_center + int(self.width/2) >= binimg.shape[1]:
            x_center = binimg.shape[1] - int(self.width/2) - 1

        return WindowBox(binimg, x_center, y_top,
                         width=self.width, height=self.height, mincount=self.mincount,
                         lane_found=self.lane_found)

    def _nonzerox_count(self):
        return len(self.nonzerox)

    def has_line(self):
        return (self._nonzerox_count() > self.mincount)

    def has_lane(self):
        if not self.lane_found and self.has_line():
            self.lane_found = True
        return self.lane_found

    def __str__(self):
        return "WindowBox [%.3f, %.3f, %.3f, %.3f]" % (self.x_left,
                                                       self.y_bottom,
                                                       self.x_right,
                                                       self.y_top)



class Line():
    """Lớp Đường (Line)"""

    # Thông báo lỗi khi đa thức của đường không hợp lệ
    NOTE_POLYNOMIAL_INVALID_MSG = "Đường có đa thức không hợp lệ."

    def __init__(self, ploty, poly_fit, binimg):
        """
        Khởi tạo đối tượng Line.

        Parameters:
        - ploty: Tọa độ y của các điểm trên đường.
        - poly_fit: Đa thức bậc 2 biểu diễn đường.
        - binimg: Ảnh nhị phân của đường.

        Raises:
        - ValueError nếu đa thức không hợp lệ.
        """
        self.__ploty = ploty
        self.__poly_fit = poly_fit
        self.__binimg = binimg

        # Các thuộc tính được tính toán từ các thuộc tính khác
        self.__curvature = None
        self.__y_bottom = np.min(ploty)
        self.__y_top = np.max(ploty)
#co dieu chinh
        try:
            # Tính toán tọa độ x ở dưới cùng và trên cùng của đường dựa trên đa thức đã cho
            self.__x_bottom = poly_fitx(self.__y_bottom, self.poly_fit)

            self.__x_top = poly_fitx(self.__y_top, self.poly_fit)

        except TypeError:
            #pass
            raise ValueError(Line.NOTE_POLYNOMIAL_INVALID_MSG)

    @property
    def xs(self):
        # Trả về tọa độ x của các điểm trên đường dựa trên đa thức
        return poly_fitx(self.ploty, self.poly_fit)

    @property
    def ploty(self):
        # Trả về tọa độ y của các điểm trên đường
        return self.__ploty

    @property
    def poly_fit(self):
        # Trả về đa thức biểu diễn đường
        return self.__poly_fit

    @property
    def binimg(self):
        # Trả về ảnh nhị phân của đường
        return self.__binimg

    @property
    def y_bottom(self):
        # Trả về tọa độ y của điểm dưới cùng trên đường
        return self.__y_bottom

    @property
    def y_top(self):
        # Trả về tọa độ y của điểm trên cùng trên đường
        return self.__y_top

    @property
    def x_bottom(self):
        # Trả về tọa độ x của điểm dưới cùng trên đường
        return self.__x_bottom

    @property
    def x_top(self):
        # Trả về tọa độ x của điểm trên cùng trên đường
        return self.__x_top

    @property
    def curvature(self):
        # Tính toán và trả về độ cong của đường
        if self.__curvature is None:
            self.__curvature = calc_curvature(self.poly_fit)
        return self.__curvature

    def __str__(self):
        # Biểu diễn đối tượng Line dưới dạng chuỗi
        return "Line(%.3f, %s, bot:(%d,%d) top:(%d,%d))" % (self.curvature,
                                                            self.poly_fit,
                                                            self.x_bottom, self.y_bottom,
                                                            self.x_top, self.y_top)


# contains the Line details and methods to determine current road line
class RoadLine():
    """RoadLine Class"""

    # Thông báo lỗi khi đường không hợp lệ
    LINE_ISNT_SANE_MSG = "Line didn't pass sanity checks."

    def __init__(self, line, poly_fit, line_history_max=6):
        """
        Khởi tạo đối tượng RoadLine.

        Parameters:
        - line: Đối tượng Line biểu diễn đường.
        - poly_fit: Mảng NumPy biểu diễn đa thức tìm được từ dữ liệu đường.
        - line_history_max: Số lượng lớn nhất của lịch sử đường được giữ.

        Attributes:
        - __line_history: Dãy lịch sử đường đã xác định.
        - __all_curvatures: Danh sách tất cả các độ cong từ lịch sử, được tạo ra lười biếng.
        - line_history_max: Số lượng lớn nhất của lịch sử đường được giữ.
        - line: Đường hiện tại.
        - poly_fit: Mảng NumPy biểu diễn đa thức tìm được từ dữ liệu đường.
        """
        self.__line_history = deque([])
        self.__all_curvatures = None
        self.line_history_max = line_history_max
        self.line = line  # phải có ít nhất một đường để bắt đầu
        self.poly_fit = poly_fit

    def __str__(self):
        """
        Biểu diễn đối tượng RoadLine dưới dạng chuỗi.
        """
        return "RoadLine(%s, poly_fit=%s, mean_fit=%s, mean_curvature=%.3f, history=[%s])" % (self.line,
                                                                                                self.poly_fit,
                                                                                                self.mean_fit,
                                                                                                self.mean_curvature,
                                                                                                ', '.join([str(l.curvature) for l in self.__line_history]))


    @property
    def line(self):
        """
        Thuộc tính line, trả về đường hiện tại từ lịch sử hoặc None nếu không có.
        """
        if len(self.__line_history) > 0:
            return self.__line_history[-1]
        else:

            return None

    @line.setter
    def line(self, line):
        """
        Thiết lập đường hiện tại và kiểm tra tính hợp lệ của đường.

        Parameters:
        - line: Đường để đặt.

        Raises:
        - ValueError nếu đường không hợp lệ dựa trên lịch sử.
        """

#khong can thiet
        def xs_are_near(x1, x2, limit=100):
            return np.abs(x1 - x2) < limit

        def is_line_sane(line):
            try:
                line_prev = self.line

                # top is near car
                if not xs_are_near(line.x_top, line_prev.x_top):
                    raise ValueError("top xs weren't near")

                # bottom is horizon
                if not xs_are_near(line.x_bottom, line_prev.x_bottom):
                    raise ValueError("bottom xs weren't near")

            # nếu đường không cắt tại y_bottom không phải là hình elip và sẽ ném ValueError
            except ValueError as err:
                print("value error: {0}".format(err))
                return False

            return True

        # Nếu có lịch sử và đường không hợp lệ, nâng lên ValueError
        if len(self.__line_history) > 0 and not is_line_sane(line):
            raise ValueError(self.LINE_ISNT_SANE_MSG)

        self._queue_to_history(line)

    def _queue_to_history(self, line):
        """
        Thêm đường vào lịch sử và giữ số lượng đường tối đa.

        Parameters:
        - line: Đường để thêm vào lịch sử.
        """
        self.__line_history.append(line)
        self.__all_curvatures = None  # sẽ được tạo lại lười biếng
        # Chỉ giữ các đường cuối cùng được thêm vào
        if self.line_history_count > self.line_history_max:
            self.__line_history.popleft()

    @property
    def line_history_count(self):
        """
        Trả về số lượng đường trong lịch sử.
        """
        return len(self.__line_history)
#co chinh sua
    @property
    def curvature(self):
        """
        Trả về độ cong của đường hiện tại.
        """
        if self.line is not None:
            return self.line.curvature
        else:
            return 0  # Hoặc giá trị mặc định tùy thuộc vào logic của bạn

    @property
    def curvatures(self):
        """
        Trả về mảng các độ cong từ lịch sử.
        """
        if self.__all_curvatures is None:
            self.__all_curvatures = np.array([line.curvature for line in self.__line_history])
        return self.__all_curvatures

    @property
    def mean_curvature(self):
        """
        Trả về độ cong trung bình từ lịch sử.
        """
        return np.mean(self.curvatures)

    @property
    def std_curvature(self):
        """
        Trả về độ cong chuẩn từ lịch sử.
        """
        return np.std(self.curvatures)

    @property
    def max_abs_curvature_variance(self):
        """
        Trả về giá trị lớn nhất của biến động tuyệt đối của độ cong từ trung bình.
        """
        return np.max(np.abs(self.curvatures - np.mean(self.curvatures)))

    @property
    def line_fits(self):
        """
        Trả về mảng các đa thức biểu diễn đường từ lịch sử.
        """
        return np.array([line.poly_fit for line in self.__line_history])

    @property
    def mean_fit(self):
        """
        Trả về đa thức biểu diễn đường trung bình từ lịch sử, với trọng số được áp dụng.
        """
        lf = self.line_fits
        nweights = len(lf)

        weights = None
        # Nếu chỉ có một đa thức thì không cần trọng số
        if nweights > 1:
            # Nếu có hai đa thức thì đưa trọng số nhiều hơn cho đa thức mới hơn
            if nweights == 2:
                weights = [.80, .20]
            # Nếu có nhiều hơn hai thì bắt đầu với hai và đệm đều phần còn lại
            else:
                weights = [.40, .30]

                # Đệm đều trọng số
                if nweights > len(weights):
                    weights = np.pad(weights, (0, nweights - len(weights)), 'constant',
                                     constant_values=(1 - np.sum(weights)) / (nweights - len(weights)))

        return np.average(lf, weights=weights, axis=0)

    @property
    def ploty(self):
        """
        Trả về tọa độ y của các điểm trên đường, tất cả đều có cùng một ploty.
        """
        return self.line.ploty

    @property
    def mean_xs(self):
        """
        Trả về tọa độ x trung bình của các điểm trên đường từ lịch sử.
        """
        return poly_fitx(self.ploty, self.mean_fit)


# contains the left & right road lines and associated image
class Lane():
    """Lane Class"""
    lane_px_width = 413
    xm_per_pix = 3.7 / lane_px_width  # meters per pixel in x dimension - copied from calc_curvature for convenience
#, mtx, dist
    def __init__(self, img_height, img_width):
        self.__img_height = img_height
        self.__img_width = img_width

        # camera callibration
        #self.__mtx = mtx
        #self.__dist = dist

        # set images to None
        self.__image = None
        self.__undistorted_image = None
        self.__binary_image = None
        self.__warped = None
        self.old_left_poly_fit = None

        self.old_right_poly_fit = None

        # warp points and transformation matrices
        warp_src, warp_dst = calc_warp_points(img_height, img_width)
        self.__warp_src = warp_src
        self.__warp_dst = warp_dst
        M, Minv = perspective_transforms(warp_src, warp_dst)
        self.__M = M
        self.__Minv = Minv

        # arrays for plotting
        warped_height = img_width
        self.__ploty = np.linspace(0, warped_height - 1, warped_height)
        self.__pts_zero = np.zeros((1, warped_height, 2), dtype=int)

        # place holders for the road lines - cant initialise until we have a camera image
        self.__road_line_left = None
        self.__road_line_right = None

    @property
    def img_height(self):
        return self.__img_height

    @property
    def img_width(self):
        return self.__img_width

    @property
    def pts_zero(self):
        return self.__pts_zero

    @property
    def ploty(self):
        return self.__ploty

    @property
    def undistorted_image(self):
        return self.__undistorted_image

    @property
    def binary_image(self):
        return self.__binary_image

    @property
    def warped(self):
        return self.__warped

    @property
    def road_line_left(self):
        return self.__road_line_left

    @road_line_left.setter
    def road_line_left(self, value):
        self.__road_line_left = value

    @property
    def road_line_right(self):
        return self.__road_line_right

    @road_line_right.setter
    def road_line_right(self, value):
        self.__road_line_right = value

    @property
    def result(self):
        result,distance = self._draw_lanes_unwarped()
        return result,distance
#co chinh sua
    @property
    def result_decorated(self):
        font = cv2.FONT_HERSHEY_SIMPLEX

        rll, rlr = self.road_line_left, self.road_line_right
        # Kiểm tra và thiết lập giá trị mặc định cho curvature nếu là None


        result = self.result


        return result
    @property
    def image(self):
        return self.__image

    def _image_yuv_equalize(self, image):
        # Cân bằng kênh Y của không gian màu YUV
        yuv = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YUV))
        yuv[0] = cv2.equalizeHist(yuv[0])
        return cv2.cvtColor(cv2.merge(yuv), cv2.COLOR_YUV2RGB)

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, image):
        # Đặt ảnh đầu vào và làm mờ nó
        self.__image = image
        undistorted=self.__image
        #undistorted = undistort_image(self.__image, self.__mtx, self.__dist)
        self.__undistorted_image = image

        # Tìm kiếm một warp không quá nhiễu
        self.__binary_image, self.__warped = self._undistort_warp_search(undistorted)

        # Lần đầu tiên, bắt đầu với việc tìm kiếm histogram đầy đủ để khởi tạo đối tượng road line trái và phải
        # Trước khi sử dụng _full_histogram_lane_search

        if self.road_line_left is None and self.road_line_right is None:
            self._full_histogram_lane_search()
        # Đối với các frame sau, tính lại sử dụng đường bậc thức đa thức trước đó
        else:
                self._recalc_road_lines_from_polyfit()

    def _undistort_warp_search(self, undistorted):
        # Ngưỡng ban đầu

        # Warp ảnh undistorted bằng ngưỡng đã chọn
        binary_image, warped = self._warp_undistorted_threshold(undistorted)
        cv2.imshow('warped',warped)

        return binary_image, warped

    def _warped_window_thresholding(self, undistorted, nwindows=12):
        # Warp ảnh undistorted mà không chỉ định ngưỡng
        binimg, warped = self._warp_undistorted_threshold(undistorted)

        # Tạo một ảnh warp trắng mới
        height = undistorted.shape[1]  # Thực sự bằng chiều rộng của ảnh không méo
        width = undistorted.shape[0]  # Chúng tôi đã quay warp 90 độ
        warp_window_stack = []

        def warp_window_threshold(wbot, wheight):
            # Trích xuất một cửa sổ từ ảnh warped
            warp_window = warped[wbot:wbot + wheight, :]
            return warp_window

        # Giả sử cửa sổ xuống và xếp chồng các warp lên nhau
        wheight = int(height / nwindows)
        for wbot in range(0, height - 1, wheight):
            warp_window = warp_window_threshold(wbot, wheight)
            warp_window_stack.append(warp_window)

        # Đối với phiên bản dự án này, chỉ sử dụng ảnh bình thường vào ban ngày
        warped = np.vstack(warp_window_stack)

        return binimg, warped

    def _warp_undistorted_threshold(self, undistorted_image):
        binary_image = combined_threshold(undistorted_image)
        warped = perspective_warp(binary_image, self.__M)


        return (binary_image, warped)
    @property
    def x_center_offset(self):
        # Tính offset của trung tâm xe so với trung tâm làn đường
        lx = self.x_start_left
        rx = self.x_start_right
        xcenter = int(self.warped.shape[1] / 2)
        offset = (rx - xcenter) - (xcenter - lx)
        return self._x_pix_to_m(offset)

    @property
    def x_start_left(self):
        # Trả về tọa độ x của điểm bắt đầu của làn đường bên trái
        return self.road_line_left.line.x_top

    @property
    def x_start_right(self):
        # Trả về tọa độ x của điểm bắt đầu của làn đường bên phải
        return self.road_line_right.line.x_top

    def _x_pix_to_m(self, pix):
        # Chuyển đổi từ đơn vị pixel sang đơn vị mét
        return pix * self.xm_per_pix

    def _full_histogram_lane_search(self):
        # Tìm kiếm làn đường sử dụng histogram đầy đủ
        histogram = self.warped_histogram
        peak_left, peak_right = lane_peaks(histogram)

        # Call _road_line_box_search to get line and poly_fit
        left_line, left_poly_fit = self._road_line_box_search(peak_left, self.old_left_poly_fit)
        right_line, right_poly_fit = self._road_line_box_search(peak_right, self.old_right_poly_fit)

        # Create RoadLine objects with line and poly_fit
        self.__road_line_left = RoadLine(left_line, left_poly_fit)
        self.__road_line_right = RoadLine(right_line, right_poly_fit)

        # Lưu giá trị của poly_fit cũ
        self.old_left_poly_fit = left_poly_fit
        self.old_right_poly_fit = right_poly_fit

    @property
    def warped_histogram(self):
        # Tính histogram của ảnh đã warp
        height = self.warped.shape[0]
        return lane_histogram(self.warped, int(150), 240)

    def _road_line_box_search(self, x_start, old, nwindows=12, width=40):
        # Tìm kiếm làn đường trong một cửa sổ trượt
        ytop = self.warped.shape[0]
        height = int(ytop / nwindows)

        wb = WindowBox(self.warped, x_start, ytop, width=width, height=height)
        boxes = find_lane_windows(wb, self.warped)
        poly_fit = calc_fit_from_boxes(boxes, old)
        line = self._line_from_fit(poly_fit)
        #line = Line(self.ploty, poly_fit, self.warped)
        return line,poly_fit

    def _recalc_road_lines_from_polyfit(self, margin=5):
        # Tính toán lại làn đường từ đường bậc thức đa thức trước đó
        left_fit = self.road_line_left.line.poly_fit
        right_fit = self.road_line_right.line.poly_fit

        new_left_fit, new_right_fit = calc_lr_fit_from_polys(self.warped, left_fit, right_fit, margin)
        try:
            self.road_line_left.line = self._line_from_fit(new_left_fit)
        except:
            print("really bad left line ... next version will fix this")

    def _line_from_fit(self, new_fit):
        # Tạo đối tượng Line từ đường bậc thức đa thức
        if new_fit is None:
            raise ValueError("no polynominal fit")

        try:
            line = Line(self.ploty, new_fit, self.warped)
        except ValueError as err:
            print("line not sane - skipping it - error: {0}".format(err))
        except TypeError as err:
            print("type error: {0}".format(err))

        return line







    def _draw_lanes_unwarped(self):
        # Vẽ làn đường trên ảnh gốc
        font = cv2.FONT_HERSHEY_SIMPLEX

        warp_zero = np.zeros_like(self.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        mean_xs_left, mean_xs_right = self.road_line_left.mean_xs, self.road_line_right.mean_xs
        pts_left = np.array([np.transpose(np.vstack([mean_xs_left, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([mean_xs_right, self.ploty])))])

        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 120,255))

        # Extract the four vertices of the filled polygon
        vertex1 = (int(pts_left[0][0][0]), int(pts_left[0][0][1]))
        #vertex2 = (int(pts_right[0][0][0]), int(pts_right[0][0][1]))
        vertex3 = (int(pts_right[0][-1][0]), int(pts_right[0][-1][1]))
        #vertex4 = (int(pts_left[0][-1][0]), int(pts_left[0][-1][1]))

        # Calculate the midpoint between vertex3 and vertex1
        midpoint_x = (vertex3[0] + vertex1[0]) // 2
        midpoint_y = (vertex3[1] + vertex1[1]) // 2
        distance = vertex3[0]-vertex1[0]

        midpoint = (midpoint_x, midpoint_y)


        # Draw a circle at the midpoint for visualization
        cv2.circle(color_warp, midpoint, 10, (0, 0, 255), -1)
        cv2.circle(color_warp, (134,100), 5, (255,0 ,0 ), -1)


        newwarp = perspective_unwarp(color_warp, self.__Minv)
        result = cv2.addWeighted(self.undistorted_image, 1, newwarp, 0.3, 0)
        # Display the distance on the image
        font_scale = 1
        font_thickness = 2
        font_color = (255, 255, 255)
        cv2.putText(result, 'Distance: %.2f meters' % distance, (50, 50),
                    font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        return result,distance










