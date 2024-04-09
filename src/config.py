import torch

ROOT_DIR = '/Users/kimsk/Documents/ocr_intern_task'
TRAIN_DIR = ROOT_DIR + '/train'
TEST_DIR = ROOT_DIR + '/test'
SAVE_PATH = ROOT_DIR + '/model/model.pt'  ## 모델 파일 저장 경로
LOG_PATH = ROOT_DIR + '/log/train_log.txt'  ## 학습 로그 파일 경로
TEST_CSV_PATH = ROOT_DIR + '/test_result.csv'  ## 추론 결과 파일 경로
BATCH_SIZE = 16
TARGET_SIZE = 6  ## 라벨 길이
TARGET_NUM = 10  ## 라벨 숫자 개수
L_RATE = 3e-4
EPOCH = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")