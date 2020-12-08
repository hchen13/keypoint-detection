from datetime import datetime

from ethan_toolbox import draw_points, cv2
from imutils.video import VideoStream

from prototype.pvanet import KeypointNet
import numpy as np

if __name__ == '__main__':
    print('[info] initializing KeypointNet model and live camera')
    tick = datetime.now()
    vs = VideoStream().start()
    net = KeypointNet(K=5)
    net.load_weights('facial_pvanet.h5')
    net.model.predict(np.random.uniform(size=(1, 256, 256, 3)))
    tock = datetime.now()
    init_time = tock - tick
    print(f"[info] complete, initialize time: {init_time.total_seconds():.2f} sec.\n")

    print('[info] start streaming...')
    while True:
        t0 = datetime.now()
        frame = vs.read()
        keypoints = net.detect(frame)
        disp = draw_points(frame, keypoints, color=(10, 200, 19), radius=3)
        disp = np.array(disp[:, ::-1, :])

        t1 = datetime.now()
        delta = (t1 - t0).total_seconds()
        fps = 1 / delta
        cv2.putText(disp, f"FPS {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_COMPLEX, .7, (10, 200, 19), thickness=1)

        cv2.imshow('live', disp)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break