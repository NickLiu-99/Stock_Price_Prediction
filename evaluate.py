from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 替换Arial Unicode MS字体

def eval():
    # model = torch.load(args.save_file)
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    global close_max
    global close_min
    global preds
    global labels
    preds = []
    labels = []
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        if args.useGPU:
            x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
        else:
            x = x.squeeze(1)
        pred = model(x)
        list = pred.data.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())

    for i in range(len(preds)):
        preds[i][0] = preds[i][0] * (close_max - close_min) + close_min
        labels[i] = labels[i] * (close_max - close_min) + close_min
        print('预测值是%.2f,真实值是%.2f' % (preds[i][0], labels[i]))
eval()


#绘制预测图
#测试集绘图
fig = plt.figure(figsize=(15,5))
axes = fig.add_subplot(1,1,1)
axes.set_title('LSTM神经网络预测图')
axes.set_xlabel('测试日期数')
axes.set_ylabel('收盘价')
axes.plot(range(len(preds)),preds,label='预测值')
axes.plot(range(len(preds)),labels,label='真实值')
axes.legend(loc='upper right')
plt.show()

# 根据真实值与测试值计算RMSE
# 使用sklearn调用衡量线性回归的MSE 、 RMSE、 MAE、r2
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


print("rmse:", sqrt(mean_squared_error(labels, preds)))