import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

""" training configs """
ENV = 'local' if os.environ.get("AIBOX") is None else 'aibox'
if ENV == 'local':
    EPOCHS = 1
    BATCH_SIZE = 4
    DATASET_ROOT = '/Users/ethan/datasets/face-keypoints'
else:
    EPOCHS = 100
    BATCH_SIZE = 64
    DATASET_ROOT = '/media/ethan/DataStorage/face-keypoints'

LEARNING_RATE = 1e-5
KEYPOINT_TYPES = 5

LOG_STEPS = 20
EXAM_STEPS = 300