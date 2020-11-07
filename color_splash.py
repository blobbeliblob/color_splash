
import numpy as np
import cv2
import os

def color_splash(img, hue_boundaries=(0, 180)):
	# boundaries
	lower_h = hue_boundaries[0]
	upper_h = hue_boundaries[1]
	# convert rgb to hsv
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# convert input to grayscale
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# define the mask
	if lower_h > upper_h:
		lower_boundary = np.array([0,50,50])
		upper_boundary = np.array([upper_h,255,255])
		mask_1 = cv2.inRange(img_hsv, lower_boundary, upper_boundary)
		lower_boundary = np.array([lower_h,50,50])
		upper_boundary = np.array([180,255,255])
		mask_2 = cv2.inRange(img_hsv, lower_boundary, upper_boundary)
		mask = mask_1 + mask_2
	else:
		lower_boundary = np.array([lower_h,50,50])
		upper_boundary = np.array([upper_h,255,255])
		mask = cv2.inRange(img_hsv, lower_boundary, upper_boundary)
	# inverted mask
	mask_inv = cv2.bitwise_not(mask)
	# convert everything except the defined range to black
	img_with_color = cv2.bitwise_or(img, img, mask = mask)
	img_without_color = cv2.bitwise_or(img_gray, img_gray, mask = mask_inv)
	img_without_color = np.stack((img_without_color,)*3, axis=-1)
	img_mod = img_with_color + img_without_color
	return img_mod


if __name__=='__main__':
	# hue boundaries
	lower_hue_str = "\nSpecify the lower hue boundary (The boundary should be between 0 and 180):\n"
	lower_hue = int(input(lower_hue_str))
	while lower_hue not in range(0, 180+1):
		print("\nNot recognised! Make sure the input is between 0 and 180\n")
		img_or_vid = int(input(lower_hue_str))
	upper_hue_str = "\nSpecify the upper hue boundary (The boundary should be between 0 and 180):\n"
	upper_hue = int(input(upper_hue_str))
	while upper_hue not in range(0, 180+1):
		print("\nNot recognised! Make sure the input is between 0 and 180\n")
		img_or_vid = int(input(upper_hue_str))
	# choose between image or video
	image_or_video_str = "\nImage or video?\n\t1 = image\n\t2 = video\n"
	img_or_vid = int(input(image_or_video_str))
	while img_or_vid not in range(1, 3):
		print("\nNot recognised! Make sure the input is either 1 or 2\n")
		img_or_vid = int(input(image_or_video_str))
	
	# image
	if img_or_vid == 1:
	
		# specify the path to the image file
		path_select_str = "\nImage path:\n"
		img_path = input(path_select_str)
		# specify the path to which the new image is saved
		save_path = os.path.splitext(img_path)[0] + '_mod' + os.path.splitext(img_path)[1]

		# load image
		img = cv2.imread(img_path, 1)
		# modify image
		img_mod = color_splash(img, (lower_hue, upper_hue))
		# save image
		cv2.imwrite(save_path, img_mod)

	# video
	elif img_or_vid == 2:

		# specify the path to the video file
		path_select_str = "\nVideo path:\n"
		video_path = input(path_select_str)
		# specify the path to which the new video is saved
		save_path = os.path.splitext(video_path)[0] + '_mod' + '.avi'

		#load video
		cap = cv2.VideoCapture(video_path)

		if not cap.isOpened():
			print("\nCOULDN'T OPEN FILE\n")
			print("\nTERMINATING")

		resize_video = False
		fps = 24.0
		cap.set(cv2.CAP_PROP_FPS, fps)
		frame_width = int(cap.get(3))
		frame_height = int(cap.get(4))
		if resize_video:
			frame_width = 640
			frame_height = 360
		fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
		out = cv2.VideoWriter(save_path,fourcc, fps, (frame_width, frame_height))

		length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		counter = 0
		while cap.isOpened():
			ret, frame = cap.read()
			if ret:
				if resize_video:
					frame = cv2.resize(frame, (frame_width, frame_height))
				# modify frame
				frame = color_splash(frame, (lower_hue, upper_hue))
				# write the frame to the output
				out.write(frame)
				# display the progress
				counter += 1
				print("Progress:\t"+str(counter)+" / "+str(length), end="\r")
				#cv2.imshow('Frame',frame)
				#if cv2.waitKey(1) & 0xFF == ord('q'):
				#	break
			else:
				break

		cap.release()
		out.release()
		print()

	print("\nTERMINATING")
