
import numpy as np
import cv2

def color_splash(img):
    # convert rgb to hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # convert input to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # define the mask
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask_1 = cv2.inRange(img_hsv, lower_red, upper_red)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask_2 = cv2.inRange(img_hsv, lower_red, upper_red)
    mask = mask_1 + mask_2
    # inverted mask
    mask_inv = cv2.bitwise_not(mask)
    # convert everything except the defined range to black
    img_with_color = cv2.bitwise_or(img, img, mask = mask)
    img_without_color = cv2.bitwise_or(img_gray, img_gray, mask = mask_inv)
    img_without_color = np.stack((img_without_color,)*3, axis=-1)
    img_mod = img_with_color + img_without_color
    return img_mod

img_path = 'idle_days.png'
save_path = 'idle_days_pop.png'

# load image
img = cv2.imread(img_path, 1)
# modify image
img_mod = color_splash(img)
# save image
cv2.imwrite(save_path, img_mod)

