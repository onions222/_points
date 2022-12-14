import os

from dataset import *
from net import *
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

path = 'test_image'
net = Net()
net.load_state_dict(torch.load('params/net.pth'))

# 这个很重要！！
net.eval()

for i in os.listdir(path):
    img = Image.open(os.path.join(path, i))
    img_data = tf(img)
    img_data = torch.unsqueeze(img_data, dim=0)
    draw = ImageDraw.Draw(img)
    # 做了归一化
    out = net(img_data)
    out = (out[0]*100).tolist()
    for j in range(0, len(out), 2):
        draw.ellipse((out[j]-2, out[j+1]-2, out[j]+2, out[j+1]-2), (255, 0, 0))
    img.show()
