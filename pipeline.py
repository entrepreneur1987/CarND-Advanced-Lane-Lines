import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import glob
import numpy as np
import os
import pickle
from PIL import Image
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg

CHESSBOARD_HEIGHT = 6
CHESSBOARD_WIDTH = 9
CHESSBOARD_SIZE = (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT)

DEBUG = False

def _get_cameral_image_paths():
	""" get all image paths	"""	
	img_paths = glob.glob('./camera_cal/*.jpg')
	return img_paths


def _get_road_image_paths():
	""" get all road image paths """
	img_paths = glob.glob('./test_images/*.jpg')
	return img_paths


def _save_image(path, data):
	""" save numpy array as image to the specified path """
	mpimg.imsave(path, data)


def _get_image_file_name(img_path):
	""" get image file name from its path """
	return img_path.rpartition('/')[-1]


def _get_grayscale_image(rgb_image):
	""" convert bgr image to grayscale """
	return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def _get_hls_image(rgb_image):
	""" convert bgr image to hls """
	return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)


def calibrate_camera(output_image_dir='./output_images/chessboard'):
	"""
		Calibrate the camera which is used to take pictures of chessboard and road condition.
		The calibration data of the first successfully undistorted chessboard image is saved
		in calibration.p and used in all the following processing of all video images. 
	"""
	chessboard_img_paths = _get_cameral_image_paths()
	objpoints = []
	imgpoints = []

	# prepares corner coordinates in undistorted chessboard image
	objp = np.zeros((CHESSBOARD_HEIGHT*CHESSBOARD_WIDTH, 3), np.float32)
	objp[:,:2] = np.mgrid[0:CHESSBOARD_WIDTH, 0:CHESSBOARD_HEIGHT].T.reshape(-1,2)

	pickle_dict = dict()

	for chessboard_img_path in chessboard_img_paths:
		# read original chessboard image and create a grayscale copy
		chessboard_img = mpimg.imread(chessboard_img_path)
		gray_img = _get_grayscale_image(chessboard_img)

		# get the image file name so we can map processed images back to the original one
		img_file_name = _get_image_file_name(chessboard_img_path)

		# find chessboard corners
		ret, corners = cv2.findChessboardCorners(gray_img, CHESSBOARD_SIZE, None)
		
		# if the corners are found successfully, draw the chessboard corners and undistort the image
		# otherwise print out the image names which fail
		if ret == True:
			# prepare the objpoints and imgpoints
			objpoints.append(objp)
			imgpoints.append(corners)

			# draw chessboard corners and save it
			chessboard_drawn_img = cv2.drawChessboardCorners(chessboard_img, CHESSBOARD_SIZE, corners, ret)
			_save_image(os.path.join(output_image_dir, 'drawn_' + img_file_name), chessboard_drawn_img)

			# undistort the chessboard image and save it
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)
			undistort = cv2.undistort(chessboard_img, mtx, dist, None, mtx)
			_save_image(os.path.join(output_image_dir, 'undistort_' + img_file_name), undistort)

			# save calibration data of the 1st image which undistorted successfully 
			print(img_file_name)
			if img_file_name == 'calibration2.jpg':
				pickle_dict['mtx'] = mtx
				pickle_dict['dist'] = dist
				pickle.dump(pickle_dict, open('./calibration.p', 'wb'))
		else:
			print('Failed to find chessboard corners of image: ' + img_file_name)


def _load_calibration_data():
	""" load calibration data from the saved pickle file """
	pickle_dict = pickle.load(open('./calibration.p', 'rb'))
	mtx = pickle_dict['mtx']
	dist = pickle_dict['dist']
	return mtx, dist


def _undistort_image(img, mtx, dist):
	""" undistort an image with the given mtx and dist values """
	undistort = cv2.undistort(img, mtx, dist, None, mtx)
	return undistort


def _graidient_color_threshold(hls):
	s_channel_img = hls[:,:,2]

	s_color_threshold = (30, 255)
	s_color_binary_out = np.zeros_like(s_channel_img)
	s_color_binary_out[(s_channel_img > s_color_threshold[0]) & (s_channel_img < s_color_threshold[1])] = 1

	s_sobel_x_threshold = (10, 90)
	sobel_x = cv2.Sobel(s_channel_img, cv2.CV_64F, 1, 0, ksize=3)
	absolute_x = np.absolute(sobel_x)
	scaled_x = np.uint8(255*absolute_x/np.max(absolute_x))
	s_sobel_x_binary_out = np.zeros_like(s_channel_img)
	s_sobel_x_binary_out[(scaled_x > s_sobel_x_threshold[0]) & (scaled_x < s_sobel_x_threshold[1])] = 1

	s_sobel_y_threshold = (10, 90)
	sobel_y = cv2.Sobel(s_channel_img, cv2.CV_64F, 0, 1, ksize=3)
	absolute_y = np.absolute(sobel_y)
	scaled_y = np.uint8(255*absolute_y/np.max(absolute_y))
	s_sobel_y_binary_out = np.zeros_like(s_channel_img)
	s_sobel_y_binary_out[(scaled_y > s_sobel_y_threshold[0]) & (scaled_y < s_sobel_y_threshold[1])] = 1

	s_mag_threshold = (20, 90)
	sobel_mag = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
	scaled_mag = np.uint8(255*sobel_mag/np.max(sobel_mag))
	s_mag_binary_out = np.zeros_like(s_channel_img)
	s_mag_binary_out[(scaled_mag > s_mag_threshold[0]) & (scaled_mag < s_mag_threshold[1])] = 1

	s_direction_threshold = (0.9, 1.3)
	direction = np.arctan2(absolute_y, absolute_x)
	s_direction_binary_out = np.zeros_like(s_channel_img)
	s_direction_binary_out[(direction > s_direction_threshold[0]) & (direction < s_direction_threshold[1])] = 1

	combined = np.zeros_like(s_channel_img)
	combined[(s_color_binary_out == 1) & (s_sobel_x_binary_out == 1) | (s_sobel_y_binary_out == 1) & (s_direction_binary_out == 1)] = 1

	_show_img(combined, 'gray')
	return combined


def _show_img(img_to_draw, cmap=None):
	if DEBUG:
		plt.imshow(img_to_draw, cmap=cmap)
		plt.show()


def _get_img_to_draw(data):
	out_img = np.dstack((data, data, data)) * 255
	return out_img


def _draw_lines(img, start_point, end_point, color=[255, 0, 0], thickness=2):
	cv2.line(img, start_point, end_point, color, thickness)


def _pick_lanes(img):
	_draw_lines(img, (585,455), (270, 680))
	_draw_lines(img, (270,680), (1075, 680))
	_draw_lines(img, (1075,680), (730, 455))
	_draw_lines(img, (730,455), (585, 455))


def perspective_transform(img, src, dst):
	M = cv2.getPerspectiveTransform(src, dst)
	img_size = (img.shape[1], img.shape[0])
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	return warped, M


def reverse_perspective_transform(img, src, dst):
	Minv = cv2.getPerspectiveTransform(dst, src)
	img_size = (img.shape[1], img.shape[0])
	unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
	return unwarped, Minv


def undistort_road_images(road_image_paths, output_image_dir):
	mtx, dist = _load_calibration_data()

	for road_image_path in road_image_paths:
		img_file_name = _get_image_file_name(road_image_path)
		road_image = mpimg.imread(road_image_path)
		undistort = _undistort_image(road_image, mtx, dist)
		_save_image(os.path.join(output_image_dir, 'undistort_' + img_file_name), undistort)


def _find_lanes(warped_img, ploty):
	histogram = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)
	out_img = np.dstack((warped_img, warped_img, warped_img))*255

	# initialize the left lane center and right lane center
	midpoint = histogram.shape[0]//2
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# get all lane points
	nonzero = warped_img.nonzero()
	nonzerox = nonzero[1]
	nonzeroy = nonzero[0]

	# initialize params related to window
	n_windows = 10
	window_height = warped_img.shape[0]/n_windows

	leftx_current = leftx_base
	rightx_current = rightx_base

	margin = 100
	minpx = 50

	left_lane_xs = []
	right_lane_xs = []

	for window_index in range(n_windows):
		# get window boundaries for both left and right lanes
		leftx_min, leftx_max = leftx_current - margin, leftx_current + margin
		rightx_min, rightx_max = rightx_current - margin, rightx_current + margin
		y_min, y_max = np.int(warped_img.shape[0] - (window_index + 1) * window_height), np.int(warped_img.shape[0] - window_index * window_height)

		# print('left - x:{}-{}, y{}-{}'.format(leftx_min, leftx_max, y_min, y_max))
		# print('right - x:{}-{}, y{}-{}'.format(rightx_min, rightx_max, y_min, y_max))
		cv2.rectangle(out_img, (leftx_min, y_min), (leftx_max, y_max), (0, 255, 0), 2)
		cv2.rectangle(out_img, (rightx_min, y_min), (rightx_max, y_max), (0, 255, 0), 2)

		# get lane points for both left and right lanes
		left_window = warped_img[leftx_min:leftx_max, y_min:y_max]
		left_x = ((nonzerox > leftx_min) & (nonzerox < leftx_max) & (nonzeroy > y_min) & (nonzeroy < y_max)).nonzero()[0]
		left_lane_xs.append(left_x)

		right_window = warped_img[rightx_min:rightx_max, y_min:y_max]
		right_x = ((nonzerox > rightx_min) & (nonzerox < rightx_max) & (nonzeroy > y_min) & (nonzeroy < y_max)).nonzero()[0]
		right_lane_xs.append(right_x)

		if len(left_x) > minpx:
			leftx_current = np.int(np.mean(nonzerox[left_x]))
		if len(right_x) > minpx:
			rightx_current = np.int(np.mean(nonzerox[right_x]))

	left_lane_xs = np.concatenate(left_lane_xs)
	right_lane_xs = np.concatenate(right_lane_xs)

	leftx = nonzerox[left_lane_xs]
	lefty = nonzeroy[left_lane_xs]
	rightx = nonzerox[right_lane_xs]
	righty = nonzeroy[right_lane_xs]

	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	left_fit_x = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fit_x = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_xs], nonzerox[left_lane_xs]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_xs], nonzerox[right_lane_xs]] = [0, 0, 255]

	plt.imshow(out_img)
	plt.plot(left_fit_x, ploty, color='yellow')
	plt.plot(right_fit_x, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	_show_img(out_img)
	return left_fit, right_fit


def _find_lanes_derive(warped_img, left_fit, right_fit, ploty):
	nonzero = warped_img.nonzero()
	nonzerox = nonzero[1]
	nonzeroy = nonzero[0]

	margin = 100
	left_x_center = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]
	left_lane_xs = ((nonzerox > left_x_center - margin) & (nonzerox < left_x_center + margin))

	right_x_center = right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]
	right_lane_xs = ((nonzerox > right_x_center - margin) & (nonzerox < right_x_center + margin))

	left_x = nonzerox[left_lane_xs]
	left_y = nonzeroy[left_lane_xs]
	right_x = nonzerox[right_lane_xs]
	right_y = nonzeroy[right_lane_xs]

	left_fit = np.polyfit(left_y, left_x, 2)
	right_fit = np.polyfit(right_y, right_x, 2)

	left_fit_x = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fit_x = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


	out_img = _get_img_to_draw(warped_img)
	window_img = np.zeros_like(out_img)

	out_img[nonzeroy[left_lane_xs], nonzerox[left_lane_xs]] = [255,0,0]
	out_img[nonzeroy[right_lane_xs], nonzerox[right_lane_xs]] = [0,0,255]

	left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	plt.imshow(result)
	plt.plot(left_fit_x, ploty, color='yellow')
	plt.plot(right_fit_x, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	_show_img(result)
	return left_fit, right_fit


def _measure_curvature(left_fit, right_fit, ploty, car_pos):
	# meters per pixel
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    left_fit_x = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit_x = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_fit_m = np.polyfit(ploty*ym_per_pix, left_fit_x*xm_per_pix, 2)
    right_fit_m = np.polyfit(ploty*ym_per_pix, right_fit_x*xm_per_pix, 2)
    y_eval = np.max(ploty)

    left_curvature = ((1 + (2*left_fit_m[0]*y_eval*ym_per_pix + left_fit_m[1])**2)**1.5)/(2*np.absolute(left_fit_m[0]))
    right_curvature = ((1 + (2*right_fit_m[0]*y_eval*ym_per_pix + right_fit_m[1])**2)**1.5)/(2*np.absolute(right_fit_m[0]))

    left_pos = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_pos = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

    distance_from_center = (car_pos - (left_pos + right_pos) / 2)*xm_per_pix
    return left_curvature, right_curvature, distance_from_center


def _draw(image, undistort, warped, left_fit_x, right_fit_x, ploty, Minv, curvature, distance_from_center):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Draw curvature and distance from center
	font = cv2.FONT_HERSHEY_SIMPLEX
	text = 'Curvature: ' + '{:04.2f}'.format(curvature) + 'm'
	cv2.putText(undistort, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)

	direction = 'left'
	if distance_from_center > 0:
		direction = 'right'
	text = '{:04.3f}'.format(np.absolute(distance_from_center)) + 'm ' + direction + ' of center'
	cv2.putText(undistort, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
	return result


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 

    @staticmethod
    def sanity_check(left_fit, right_fit, ploty, left_curvature, right_curvature):
  		# check distance between two lanes
        y_eval = np.max(ploty)
        left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        if right_x - left_x > 800 or right_x - left_x < 600:
            return False

        # check cavature
        if np.absolute(left_curvature - right_curvature) > 1000:
        	return False
        return True

    def update(self, detected, fit, curvature):
    	self.detected = detected
    	if detected:
    		self.current_fit.append(fit)
    		if len(self.current_fit) > 5:
    			self.current_fit = self.current_fit[1:]
    		self.best_fit = np.mean(np.asarray(self.current_fit), axis=0)
    		self.radius_of_curvature = curvature

left_line = Line()
right_line = Line()


def process_image(img):
	# load camera calibration data
	mtx, dist = _load_calibration_data()

	# undistort image
	undistort = _undistort_image(img, mtx, dist)

	_show_img(undistort)

	# process the undistort image using graidient and color threshold
	hls = _get_hls_image(undistort)
	thresholded_img = _graidient_color_threshold(hls)

	# draw picked lanes
	img_to_draw = _get_img_to_draw(thresholded_img)
	_pick_lanes(img_to_draw)
	
	src = np.float32([[585,455], [250, 680], [1050, 680], [700, 455]])
	dst = np.float32([[300, 0], [300, 720], [1000, 720], [1000, 0]])

	# warp and unwarp image to get M and Minv
	warped_img, M = perspective_transform(thresholded_img, src, dst)
	unwarped_img, Minv = reverse_perspective_transform(warped_img, src, dst)
	_show_img(warped_img, 'gray')

	ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
	# if previous frame is detected successfully, derive this frame
	# otherwise, reset and recalculate
	if left_line.detected and right_line.detected:
		left_fit, right_fit = _find_lanes_derive(warped_img, left_line.best_fit, right_line.best_fit, ploty)
		print("Derive")
	else:
		left_fit, right_fit = _find_lanes(warped_img, ploty)
		print("Calculate")
	left_curvature, right_curvature, distance_from_center = _measure_curvature(left_fit, right_fit, ploty, undistort.shape[1]/2)

	# if sanity check passed, update left and right lines
	if Line.sanity_check(left_fit, right_fit, ploty, left_curvature, right_curvature):
		left_line.update(detected=True, fit=left_fit, curvature=left_curvature)
		right_line.update(detected=True, fit=right_fit, curvature=right_curvature)
	# otherwise use the average polyfit
	else:
		left_fit = left_line.best_fit
		right_fit = right_line.best_fit
		left_line.update(detected=False, fit=left_fit, curvature=left_curvature)
		right_line.update(detected=False, fit=right_fit, curvature=right_curvature)

	if left_fit is not None and right_fit is not None:
		left_fit_x = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fit_x = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		final_img = _draw(img, undistort, warped_img, left_fit_x, right_fit_x, ploty, Minv, (left_curvature + right_curvature)/2.0, distance_from_center)
		_show_img(final_img)
		return final_img
	return undistort

def run():
	#calibrate_camera()
	#undistort_road_images(_get_road_image_paths(), './output_images/test')
	for img_path in _get_road_image_paths():
		img = mpimg.imread(img_path)
		process_image(img)
	# process_image('./test_images/test4.jpg')

def video():
	white_output = './project_video_output2.mp4'
	clip1 = VideoFileClip("./project_video.mp4")
	white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	white_clip.write_videofile(white_output, audio=False)

if __name__ == '__main__':
	# run()
	video()
