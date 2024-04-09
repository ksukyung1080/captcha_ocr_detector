import config
from dataset import CaptchaDataset, AugmentCaptchaDataset
from model import CaptchaModel
from trainer import Trainer
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import pandas as pd

## 데이터 증강 없이 학습
def train():
    img_list = glob(config.TRAIN_DIR + '/*.png')
    data_set = CaptchaDataset(img_list)
    train_set, valid_set = train_test_split(data_set, test_size=0.2, shuffle=True, random_state=42)
    train_loader = DataLoader(train_set, batch_size = config.BATCH_SIZE)
    valid_loader = DataLoader(valid_set, batch_size = config.BATCH_SIZE)
    
    model = CaptchaModel(num_chars=config.TARGET_NUM)
    trainer = Trainer(model)
    trainer.run_train(train_loader, valid_loader, save=True)
    
## 데이터 증강 후 학습
def train_augment():
    img_list = glob(config.TRAIN_DIR + '/*.png')
    data_set = CaptchaDataset(img_list)
    
    for i in range(6,10):
        i_img_list = glob(config.TRAIN_DIR + f'/*{str(i)}*.png')[:10] ## 9-6 숫자를 포함한 이미지 파일 10개씩만 증강
        aug_data_set = AugmentCaptchaDataset(i_img_list, i)
        data_set = data_set + aug_data_set ## 증강하지 않은 기존 데이터셋에 추가
    
    train_set, valid_set = train_test_split(data_set, test_size=0.2, shuffle=True, random_state=42)
    train_loader = DataLoader(train_set, batch_size = config.BATCH_SIZE)
    valid_loader = DataLoader(valid_set, batch_size = config.BATCH_SIZE)
    
    model = CaptchaModel(num_chars=config.TARGET_NUM)
    trainer = Trainer(model)
    trainer.run_train(train_loader, valid_loader, save=True)
    
## test 파일로 추론
def test():
    # img_list = glob(config.TEST_DIR + '/[0-5]*.png')  ## 첫번째 숫자가 0-5인 이미지만 추론
    img_list = glob(config.TEST_DIR + '/*.png')
    test_set = CaptchaDataset(img_list)
    test_loader = DataLoader(test_set, batch_size = config.BATCH_SIZE)
    
    model = torch.load(config.SAVE_PATH)
    trainer = Trainer(model)
    trainer.test(test_loader, save=True)
    
if __name__ == "__main__":
    # train()
    # train_augment()
    test()