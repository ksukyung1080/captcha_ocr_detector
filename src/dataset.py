import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

## Captcha 데이터 객체
class CaptchaDataset(Dataset):
    def __init__(self, file_list): ## file_list: png 이미지 path의 리스트
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]

        ## 숫자를 인식하는데 색상이 중요하지 않고 연산량을 줄일 수 있어 Grayscale로 변환
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = transforms.ToTensor()(img)

        label = img_path.split('/')[-1].split('.')[0] ## 파일명 -> label
        label_list = torch.tensor([int(x) for x in label]) ## 6자리 각각 인식해야하므로 label을 하나씩 뜯어줌

        return img, label_list
    
## 첫번째 숫자를 6~9로 바꾸기 위한 객체
## 6-9 숫자가 시작하는 부분에서 잘라서 6-9 숫자가 앞에 오도록 이미지를 잘라서 이어붙임
class AugmentCaptchaDataset(CaptchaDataset):
    def __init__(self, file_list, start_num): ## start_num: 첫번째 숫자로 만들고 싶은 숫자 (6-9) -> start_num이 포함된 데이터가 들어와야함.
        self.file_list = file_list
        self.start_num = start_num
        
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        
        label = img_path.split('/')[-1].split('.')[0]
        num_idx = label.find(str(self.start_num))  ## start_num이 존재하는 위치
        new_label = label[num_idx:] + label[:num_idx]  ## start_num에서 시작하도록 라벨 재조정
        new_label_list = torch.tensor([int(x) for x in new_label])
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        separate_width = int(120 / 6 * num_idx) ## 이미지상에서 start_num이 시작하는 위치
        front_image = img[:, separate_width:]  ## start_num 시작 ~ 이미지 끝 부분
        back_image = img[:, :separate_width]  ## 이미지 시작 ~ start_num 시작 부분
        new_img = np.concatenate([front_image, back_image], axis=1)  ## 자른 이미지 병합
        new_img = transforms.ToTensor()(new_img)
        
        return new_img, new_label_list
        