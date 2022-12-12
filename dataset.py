import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image

# 归一化
tf = transforms.compose(
    transforms.ToTensor()
)


class my_dataset(Dataset):
    def __init__(self, root):
        f = open(root, 'r')
        self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img_path = data.split(' ')[0]
        img_data = Image.open(img_path)
        points = data.split(' ')[1:-2]
        # 图像大小为100x100，此处是对点进行归一化
        points = [int(i)/100 for i in points]

        return tf(img_data), torch.Tensor(points)


if __name__ == '__main__':
    data = my_dataset('dataset_center.txt')
    for i in data:
        print(i[0].shape)
        print(i[1])
