import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
import pandas as pd

class Trainer():
    def __init__(self, model):
        self.model = model.to(config.DEVICE)
        self.epoch = config.EPOCH
        self.device = config.DEVICE
        self.optimizer = optim.Adam(self.model.parameters(), lr= config.L_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, factor=0.8, patience=5)
    
    ## 학습 cycle 실행: 각 epoch 별로 train/validation 반복
    def run_train(self, train_loader, valid_loader, save=False):
        f = open(config.LOG_PATH, 'w') ## 학습 Log 저장
        for epoch in range(self.epoch):
            print("[ EPOCH {} ]".format(epoch+1))
            f.write("\n[ EPOCH {} ]".format(epoch+1))
            train_loss, train_log = self.train(train_loader)
            valid_loss, valid_log = self.evaluate(valid_loader)
            self.scheduler.step(valid_loss)
            
            f.write(train_log)
            f.write(valid_log)
        
        if save is True:
            torch.save(self.model, config.SAVE_PATH)
            
        f.close()
    
    ## 모델 학습
    def train(self, train_loader):
        self.model.train()
        train_log = ""
        for batch_idx, (img, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            img = img.to(self.device)
            target = target.to(self.device)
            
            out = self.model(img)
            loss = self.loss_fn(out, target)
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            
            if batch_idx % 10 == 0:
                print("[ Train batch {} ] - loss {:.4f}".format(batch_idx, loss))
                train_log += "\n[ Train batch {} ] - loss {:.4f}".format(batch_idx, loss)
                
        return loss, train_log
    
    ## Validation    
    def evaluate(self, valid_loader):
        self.model.eval()
        loss_sum = 0
        correct = 0
        with torch.no_grad():
            for img, target in valid_loader:
                img = img.to(self.device)
                target = target.to(self.device)
                
                out = self.model(img)
                
                loss = self.loss_fn(out, target)
                loss_sum += loss
                
                correct += self.correct_fn(out, target)
        loss_avg = loss_sum / len(valid_loader)
        accuracy = correct / len(valid_loader.dataset) * 100
    
        print("[ Validation ] - loss {:.4f} Accuracy: {:.2f}%".format(loss_avg, accuracy))
        valid_log = "\n[ Validation ] - loss {:.4f} Accuracy: {:.2f}%".format(loss_avg, accuracy)
        return loss_avg, valid_log
    
    ## 모델 추론
    def test(self, test_loader, save=False):
        self.model.eval()
        test_result = []
        correct_sum = 0
        with torch.no_grad():
            for img, target in test_loader:
                out = self.model(img)
                    
                prediction = self.decode_prediction(out)
                
                for p, t in zip(prediction, target):
                    t = t.tolist()
                    correct = (p == t)
                    correct_sum += correct
                    result = [p, t, correct]
                    test_result.append(result)
        accuracy = correct_sum / len(test_loader.dataset) * 100
        print("[ TEST ] - Accuracy: {:.2f}%".format(accuracy))
        
        if save is True:
            test_result_df = pd.DataFrame(test_result, columns=['PREDICTION','TARGET', 'CORRECT'])
            test_result_df.to_csv(config.TEST_CSV_PATH, sep='\t')
    
    ## CTCLoss 계산
    def loss_fn(self, out, target):
        log_probs = F.log_softmax(out, 2)
        input_length = torch.full(size=(out.size(1),), fill_value=out.size(0), dtype=torch.int32)
        target_length = torch.full(size=(out.size(1),), fill_value=config.TARGET_SIZE, dtype=torch.int32)
        loss = nn.CTCLoss(blank=10)( ## 0~9 예측 숫자, 10 -> blank
            log_probs, target, input_length, target_length
        )
        return loss

    ## 출력 결과에서 BLANK와 중복 제거
    def remove_duplicates(self, text):
        if len(text) > 1:
            i = 0
            while(i < len(text) and text[i]==10): ## 앞쪽 BLANK 제거
                i += 1
            if i == len(text): ## ALL BLANK
                return [10]
            else:
                remove_text = [text[i]] + [p for idx, p in enumerate(text[i+1:], start=i+1) if text[idx] != text[idx-1] and p!=10]
        elif len(text) == 1:
            remove_text = [text[0]]
        else:
            return ""
        return remove_text

    ## prediction 추출
    def decode_prediction(self, out):
        ## out 텐서 decode
        pred = out.permute(1,0,2)
        pred = torch.softmax(pred, 2)
        pred = torch.argmax(pred, 2)
        pred = pred.detach().cpu().numpy()

        ## 중복 제거된 최종 prediction 리스트
        new_pred = []
        for p in pred:
            p = p.tolist()
            new_pred.append(self.remove_duplicates(p)) ## 중복 제거

        return new_pred

    ## target과 비교해서 맞은 개수 카운트 후 반환
    def correct_fn(self, out, target):
        pred = self.decode_prediction(out)

        correct_count = 0
        for p, t in zip(pred, target):
            t = t.tolist()
            if p == t:
                correct_count += 1
        return correct_count