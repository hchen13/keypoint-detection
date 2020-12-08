import os
import sys
from datetime import datetime
from glob import glob
from signal import signal, SIGINT

import cv2
from ethan_toolbox import draw_points, show_image
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dataset_management.tools import load_dataset, augment
from prototype.pvanet import KeypointNet
from training.computations import train_on_batch, examine
from training.tensorboard_monitor import Monitor

import tensorflow as tf
import settings
import numpy as np


def visualize_detection(net, monitor, step):
    test_images = glob(os.path.join(settings.PROJECT_ROOT, 'tests', '*.jpg'))
    for i, path in enumerate(test_images):
        image = cv2.imread(path)
        keypoints = net.detect(image)
        disp = draw_points(cv2.resize(image, (256, 256)), keypoints, radius=4, color=(10, 220, 20))

        for j, p in enumerate(keypoints):
            cv2.putText(disp, f"{j + 1}", tuple(p + 2), cv2.FONT_HERSHEY_COMPLEX, .7, (10, 220, 20), thickness=1)

        show = np.expand_dims(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), axis=0)
        monitor.image(f'detections/{i + 1}', show, step)


def visualize_heatmaps(net, monitor, step):
    test = os.path.join(settings.PROJECT_ROOT, 'tests', '2.jpg')
    image = cv2.imread(test)
    image = cv2.resize(image, (256, 256))
    feed = np.expand_dims(image / 127.5 - 1., axis=0)
    h_pred, o_pred, identity = net.model.predict(feed)
    h_pred = h_pred[0]
    h_pred = np.where(h_pred < .5, .1, h_pred)
    for k in range(5):
        disp = image.copy().astype('float32')
        disp[..., :] *= h_pred[..., [k]].astype('float32')
        disp = np.clip(disp, 0, 255).astype('uint8')
        show = np.expand_dims(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), axis=0)
        monitor.image(f'heatmaps/{k + 1}', show, step)


def exit_handler(signal, frame):
    print("[info] exiting program, saving model weights..")
    p = os.path.join(settings.PROJECT_ROOT, 'weights', 'tmp')
    os.makedirs(p, exist_ok=True)
    p = os.path.join(p, 'killed.h5')
    net.model.save_weights(p)
    print('[info] terminated.')
    exit(0)


if __name__ == '__main__':
    _continue_train = True

    trainset, validset = load_dataset(
        settings.DATASET_ROOT, batch_size=settings.BATCH_SIZE,
        shuffle=True, augment=augment)

    print("[info] construct KeypointNet model...")
    net = KeypointNet(K=settings.KEYPOINT_TYPES)
    net.init_pvanet(os.path.join(settings.PROJECT_ROOT, 'weights', 'pvanet_init.h5'))

    if _continue_train:
        print("[info] continue training using previously saved weights.")
        p = os.path.join(settings.PROJECT_ROOT, 'weights', 'tmp', 'pause.h5')
        net.model.load_weights(p, by_name=True)
    else:
        print("[info] initialize model backbone with pre-trained PVANet weights.")
        net.init_pvanet()
    print(f"[info] model has {len(net.model.trainable_variables)} trainable weights.")
    print('[info] complete.\n')

    experiment_name = f'KeypointNet-pva@{datetime.now().strftime("%-y%m%d-%H:%M:%S")}'
    monitor = Monitor(experiment_name)

    print("[info] preparing learning policy...")
    opt = tf.keras.optimizers.Adam(learning_rate=settings.LEARNING_RATE)
    print("[info] complete.\n")

    signal(SIGINT, exit_handler)
    print('[info] strat training...')
    global_step = 0
    for e in range(settings.EPOCHS):
        print(f"epoch #{e + 1}/{settings.EPOCHS}@{datetime.now()}:")
        for local_step, batch_data in tqdm(enumerate(trainset)):
            global_step += 1
            train_report = train_on_batch(net.model, batch_data, opt)

            if global_step % settings.LOG_STEPS == 0:
                monitor.write_reports(train_report, global_step, prefix='train_')
                monitor.write_weights(net.model, step=global_step)

            if global_step % settings.EXAM_STEPS == 0:
                print("\r[info] examining...", end='', flush=True)
                exam_report = examine(net.model, validset, limit=global_step // 100)
                monitor.write_reports(exam_report, global_step, prefix='valid_')
                print("\r[info] examine complete.", end='', flush=True)

        visualize_detection(net, monitor, e + 1)
        visualize_heatmaps(net, monitor, e + 1)
        print(f"[info] saving intermediate weights at epoch #{e + 1}")
        p = os.path.join(settings.PROJECT_ROOT, 'weights', 'tmp')
        os.makedirs(p, exist_ok=True)
        net.model.save_weights(os.path.join(p, f'tmp_e{e+1}.h5'))