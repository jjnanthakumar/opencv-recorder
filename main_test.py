from PIL import ImageGrab,Image
import numpy as np
import cv2
from win32api import GetSystemMetrics
from datetime import datetime

def rect_with_rounded_corners(image, r, t, c):
    """
    :param image: image as NumPy array
    :param r: radius of rounded corners
    :param t: thickness of border
    :param c: color of border
    :return: new image as NumPy array with rounded corners
    """

    c += (255, )

    h, w = image.shape[:2]

    # Create new image (three-channel hardcoded here...)
    new_image = np.ones((h+2*t, w+2*t, 4), np.uint8) * 255
    new_image[:, :, 3] = 0

    # Draw four rounded corners
    new_image = cv2.ellipse(new_image, (int(r+t/2), int(r+t/2)), (r, r), 180, 0, 90, c, t)
    new_image = cv2.ellipse(new_image, (int(w-r+3*t/2-1), int(r+t/2)), (r, r), 270, 0, 90, c, t)
    new_image = cv2.ellipse(new_image, (int(r+t/2), int(h-r+3*t/2-1)), (r, r), 90, 0, 90, c, t)
    new_image = cv2.ellipse(new_image, (int(w-r+3*t/2-1), int(h-r+3*t/2-1)), (r, r), 0, 0, 90, c, t)

    # Draw four edges
    new_image = cv2.line(new_image, (int(r+t/2), int(t/2)), (int(w-r+3*t/2-1), int(t/2)), c, t)
    new_image = cv2.line(new_image, (int(t/2), int(r+t/2)), (int(t/2), int(h-r+3*t/2)), c, t)
    new_image = cv2.line(new_image, (int(r+t/2), int(h+3*t/2)), (int(w-r+3*t/2-1), int(h+3*t/2)), c, t)
    new_image = cv2.line(new_image, (int(w+3*t/2), int(r+t/2)), (int(w+3*t/2), int(h-r+3*t/2)), c, t)

    # Generate masks for proper blending
    mask = new_image[:, :, 3].copy()
    mask = cv2.floodFill(mask, None, (int(w/2+t), int(h/2+t)), 128)[1]
    mask[mask != 128] = 0
    mask[mask == 128] = 1
    mask = np.stack((mask, mask, mask), axis=2)

    # Blend images
    temp = np.zeros_like(new_image[:, :, :3])
    temp[(t-1):(h+t-1), (t-1):(w+t-1)] = image.copy()
    new_image[:, :, :3] = new_image[:, :, :3] * (1 - mask) + temp * mask

    # Set proper alpha channel in new image
    temp = new_image[:, :, 3].copy()
    new_image[:, :, 3] = cv2.floodFill(temp, None, (int(w/2+t), int(h/2+t)), 255)[1]

    return new_image


w, h = GetSystemMetrics(0), GetSystemMetrics(1)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
video = cv2.VideoWriter(f'output {timestamp}.mp4', fourcc, 20.0, (w, h))
camera = cv2.VideoCapture(0)
while True:
    img = ImageGrab.grab(bbox=(0, 0, w, h))
    _, frame = camera.read()
    frame = cv2.resize(frame,(150,150),interpolation=cv2.INTER_AREA)
    frame1 = rect_with_rounded_corners(np.array(frame),50,1,(0,0,255))
    frame1 = cv2.cvtColor(cv2.cvtColor(np.array(frame1),cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2RGB)
    real = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    fr_height, fr_width, _ = frame1.shape
    real[:fr_height, :fr_width, :] = frame1[:fr_height, :fr_width, :]
    cv2.imshow("Secret Capture", real)
    video.write(real)
    k = cv2.waitKey(10)
    if k == 27:
        break

cv2.destroyAllWindows()
camera.release()
