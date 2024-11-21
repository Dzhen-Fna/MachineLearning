import torch
import torch.nn as nn
from model import *
from utils import Conf_
from tqdm import tqdm
from data import get_data
from time import time


def main_process(conf):
    model = get_model(conf)


    data_train_loader,data_valid_loader,data_test_loader = get_data(conf)
    criterion, optimizer = get_criterionAndopti(conf,model.parameters())

    for epoch in range(conf.epochs):
        
        train_loss = 0
        train_num = 0
        train_corrects = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(data_train_loader)):
            start = time() if batch_idx == 0 else start
            if conf.use_cuda:
                images, labels = images.cuda(), labels.cuda()
            model.train()
            output = model(images)
            pre_lable = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = time()
            train_loss += loss.item() * images.size(0)
            train_corrects += torch.sum(pre_lable == labels.data)
            train_num += images.size(0)
            
        conf.appendTrainAcc(train_corrects.double().item() / train_num)
        conf.appendTrainLoss(train_loss / train_num)
        conf.appendTrainTime(end - start)
        conf.showTrainStatus()

        
        valid_loss = 0
        valid_num = 0
        valid_corrects = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(data_valid_loader)):
            start = time() if batch_idx == 0 else start
            if conf.use_cuda:
                images, labels = images.cuda(), labels.cuda()
            model.eval()
            output = model(images)
            pre_lable = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            end = time()
            valid_loss += loss.item() * images.size(0)
            valid_corrects += torch.sum(pre_lable == labels.data)
            valid_num += images.size(0)
        
        conf.appendValidAcc(valid_corrects.double().item() / valid_num)
        conf.appendValidLoss(valid_loss / valid_num)
        conf.appendValidTime(end - start)
        conf.showValidStatus()
        
        test_corrects = 0.0
        test_num = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(data_test_loader)):
                start = time() if batch_idx == 0 else start
                if conf.use_cuda:
                    images, labels = images.cuda(), labels.cuda()
                model.eval()
                output = model(images)
                pre_lable = torch.argmax(output, dim=1)
                test_corrects += torch.sum(pre_lable == labels.data)
                test_num += images.size(0)
                end = time()

        conf.appendTestAcc(test_corrects.double().item() / test_num)
        conf.appendTestTime(end - start)
        conf.showTestStatus()

    conf.saveCSV('./CSV/2024_10_22/')
                
if __name__ == '__main__':
    conf = Conf_(
    lr=0.01,
    use_cuda=True,
    model_name='MLPNet',
    data_name='MNIST',
    epochs=5,
    batch_size=100,
    ratio=0.8)
    main_process(conf)




