import os
from datetime import datetime

import cv2
from ethan_toolbox import draw_points, show_image

from prototype.pvanet import KeypointNet
from settings import PROJECT_ROOT

if __name__ == '__main__':
    tick = datetime.now()
    net = KeypointNet(K=5)
    net.load_weights('facial_pvanet.h5')
    tock = datetime.now()
    init_time = tock - tick
    print(f"[info] initialize time: {init_time.total_seconds():.2f} sec.")

    test_image = cv2.imread(os.path.join(PROJECT_ROOT, 'tests', '4.jpg'))
    tick = datetime.now()
    n = 10
    for _ in range(n):
        keypoints = net.detect(test_image)
    tock = datetime.now()
    inference_time = (tock - tick) / 10
    print(f"[info] inference time: {inference_time.total_seconds():.2f} sec.")

    disp = draw_points(test_image, keypoints, color=(10, 20, 222), radius=2)
    show_image(disp)





