from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parser_my import args
from dataset import getData

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 替换Arial Unicode MS字体

def train():

    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers ,
                 output_size=1, dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数为均方误差
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.0001

    close_max, close_min, train_loader, test_loader = getData(args.corpusFile,args.sequence_length,args.batch_size )
    global record
    record={}
    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data1 = data.squeeze(1).cuda()
                pred = model(Variable(data1).cuda())
                # print(pred.shape)
                pred = pred[1,:,:]
                label = label.unsqueeze(1).cuda()
                # print(label.shape)
            else:
                data1 = data.squeeze(1)
                pred = model(Variable(data1))
                pred = pred[1, :, :]
                label = label.unsqueeze(1)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss=loss.item()
        record[i]=loss
        print('total_loss为%f' % total_loss)
        if (i+1) % 10 == 0:
            # torch.save(model, args.save_file)
            torch.save({'state_dict': model.state_dict()}, args.save_file)
            print('前%d epoch，保存模型' % (i+1))

    # torch.save(model, args.save_file)
    torch.save({'state_dict': model.state_dict()}, args.save_file)

train()

#绘制图像
fig = plt.figure(figsize=(15,5))
axes = fig.add_subplot(1,1,1)
axes.set_title('LSTM神经网络训练图')
axes.set_xlabel('Epoch')
axes.set_ylabel('Loss')
axes.plot(list(record.keys()),list(record.values()))
plt.show()
