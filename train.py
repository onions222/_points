import os.path

import torch
from torch import nn, optim
from dataset import *
from net import *
from torch.utils.data import DataLoader



if __name__ == '__main__':
    device = torch.device('mps')
    net = Net().to(device)
    weights = 'params/net.pth'
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('loading Successfully!')

    # 加在优化器
    opt = optim.Adam(net.parameters())
    # 定义损失MSE
    loss_fun = nn.MSELoss()
    # 加在数据集
    dataset = MyDataset('data_center.txt')
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    epoch = 1
    while True:
        for i, (image, label) in enumerate(data_loader):
            image, label = image.to(device), label.to(device)
            # print(image.shape, label.shape)

            # 数据进入网络
            out = net(image)
            train_loss = loss_fun(out, label)
            print(f"{epoch}--{i}--train_loss{train_loss.item()}")

            # 运行三步走：清空梯度，回传以及step
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # 保存权重
        if epoch % 10 == 0:
            torch.save(net.state_dict(), 'params/net.pth')
            print('Save successfully')

        epoch += 1


