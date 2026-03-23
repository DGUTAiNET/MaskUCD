import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from scipy.io import loadmat
from torchvision.transforms import functional as F

def read_img(image_path):
    with Image.open(image_path) as img:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode not in ['RGB', 'L']:
            img = img.convert('L')
        img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)

    is_redundant = False
    if img_array.shape[-1] == 3:
        channel_diff = np.stack([
            np.allclose(img_array[..., 0], img_array[..., 1], atol=1),
            np.allclose(img_array[..., 1], img_array[..., 2], atol=1),
            np.allclose(img_array[..., 0], img_array[..., 2], atol=1)
        ])
        is_redundant = np.all(channel_diff)

    if is_redundant:
        img_array = img_array[..., 0][..., np.newaxis]

    min_val = img_array.min().astype(float)
    max_val = img_array.max().astype(float)
    normalized = (img_array.astype(float) - min_val) / (max_val - min_val)

    tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
    return tensor


def _Italy():
    _t1 = read_img('./Dataset/Italy_t1.bmp')
    _t2 = read_img('./Dataset/Italy_t2.bmp')
    _gt = Image.open('./Dataset/Italy_gt.bmp').convert('1')
    _gt = transforms.ToTensor()(_gt)
    change_pixels = _gt.sum()
    total_pixels = _gt.numel()
    print(change_pixels / total_pixels)

    _t1 = torch.log1p(_t1)
    _t1 = (_t1 - _t1.min()) / (_t1.max() - _t1.min())
    return _t1, _t2, _gt


def _Gloucester():
    _t1 = read_img('./Dataset/Gloucester_t1.png')
    _t2 = read_img('./Dataset/Gloucester_t2.png')
    _gt = Image.open('./Dataset/Gloucester_gt.png').convert('1')
    _gt = transforms.ToTensor()(_gt)
    change_pixels = _gt.sum()
    total_pixels = _gt.numel()
    print(change_pixels / total_pixels)
    _t2 = torch.log1p(_t2)
    _t2 = (_t2 - _t2.min()) / (_t2.max() - _t2.min())
    return _t1, _t2, _gt


def _California():
    _t1 = read_img('./Dataset/California_t1.bmp')
    _t2 = read_img('./Dataset/California_t2.bmp')
    _gt = Image.open('./Dataset/California_gt.bmp').convert('1')
    _gt = transforms.ToTensor()(_gt)
    change_pixels = _gt.sum()
    total_pixels = _gt.numel()
    print(change_pixels / total_pixels)
    scale = 0.6
    _, H, W = _t1.shape
    new_H = int(H * scale)
    new_W = int(W * scale)

    # 4. 对三张图像统一缩小
    _t1 = F.resize(_t1, [new_H, new_W], interpolation=transforms.InterpolationMode.BILINEAR)
    _t2 = F.resize(_t2, [new_H, new_W], interpolation=transforms.InterpolationMode.BILINEAR)
    _gt = F.resize(_gt, [new_H, new_W], interpolation=transforms.InterpolationMode.NEAREST)  # GT 用 NEAREST 防止灰度化
    _t2 = torch.log1p(_t2)
    _t2 = (_t2 - _t2.min()) / (_t2.max() - _t2.min())
    return _t1, _t2, _gt



def _Ottawa():
    _t1 = read_img('./Dataset/Ottawa_t1.png')
    _t2 = read_img('./Dataset/Ottawa_t2.png')
    _gt = Image.open('./Dataset/Ottawa_gt.png').convert('1')
    _gt = transforms.ToTensor()(_gt)
    change_pixels = _gt.sum()
    total_pixels = _gt.numel()
    print(change_pixels / total_pixels)
    _t2 = torch.log1p(_t2)
    _t2 = (_t2 - _t2.min()) / (_t2.max() - _t2.min())
    _t1 = torch.log1p(_t1)
    _t1 = (_t1 - _t1.min()) / (_t1.max() - _t1.min())
    return _t2, _t1, _gt


def _YR1():
    _t1 = read_img('./Dataset/YR1_t1.bmp')
    _t2 = read_img('./Dataset/YR1_t2.bmp')
    _gt = Image.open('./Dataset/YR1_gt.bmp').convert('1')
    _gt = transforms.ToTensor()(_gt)
    change_pixels = _gt.sum()
    total_pixels = _gt.numel()
    print(change_pixels / total_pixels)
    _t2 = torch.log1p(_t2)
    _t2 = (_t2 - _t2.min()) / (_t2.max() - _t2.min())
    _t1 = torch.log1p(_t1)
    _t1 = (_t1 - _t1.min()) / (_t1.max() - _t1.min())
    return _t2, _t1, _gt


def _Texas():
    mat = loadmat('./Dataset/#2-Texas-L8.mat')
    _t1 = np.array(mat['image_t1'], dtype=np.float64)
    _t1 = transforms.ToTensor()(_t1)
    _t1 = (_t1 - _t1.min()) / (_t1.max() - _t1.min())
    _t1 = _t1.to(torch.float)
    _t2 = np.array(mat['image_t2'], dtype=np.float64)
    _t2 = transforms.ToTensor()(_t2)
    _t2 = (_t2 - _t2.min()) / (_t2.max() - _t2.min())
    _t2 = _t2.to(torch.float)
    _gt = np.array(mat['Ref_gt'], dtype=np.float64)
    _gt = transforms.ToTensor()(_gt).to(torch.float)
    change_pixels = _gt.sum()
    total_pixels = _gt.numel()
    print(change_pixels / total_pixels)
    scale = 0.5
    _, H, W = _t1.shape
    new_H = int(H * scale)
    new_W = int(W * scale)

    _t1 = F.resize(_t1, [new_H, new_W], interpolation=transforms.InterpolationMode.BILINEAR)
    _t2 = F.resize(_t2, [new_H, new_W], interpolation=transforms.InterpolationMode.BILINEAR)
    _gt = F.resize(_gt, [new_H, new_W], interpolation=transforms.InterpolationMode.NEAREST)

    # _t1 = _t1[:, :new_H//2, :]
    # _t2 = _t2[:, :new_H//2, :]
    # _gt = _gt[:, :new_H//2, :]
    #
    # left = new_W // 4
    # right = new_W - new_W // 4
    #
    # _t1 = _t1[:, :, left:right]
    # _t2 = _t2[:, :, left:right]
    # _gt = _gt[:, :, left:right]
    #
    # change_pixels = _gt.sum()
    # total_pixels = _gt.numel()
    # print(change_pixels / total_pixels)
    #plt.imsave('./2.png', _gt.squeeze().cpu().numpy(), cmap='gray')
    return _t1, _t2, _gt



def _YR3():
    _t1 = read_img('./Dataset/YR3_t1.bmp')
    _t2 = read_img('./Dataset/YR3_t2.bmp')
    _gt = Image.open('./Dataset/YR3_gt.bmp').convert('1')
    _gt = transforms.ToTensor()(_gt)
    change_pixels = _gt.sum()
    total_pixels = _gt.numel()
    print(change_pixels / total_pixels)
    _t2 = torch.log1p(_t2)
    _t2 = (_t2 - _t2.min()) / (_t2.max() - _t2.min())
    _t1 = torch.log1p(_t1)
    _t1 = (_t1 - _t1.min()) / (_t1.max() - _t1.min())
    return _t2, _t1, _gt


def _YR2():
    _t1 = read_img('./Dataset/YR2_t1.bmp')
    _t2 = read_img('./Dataset/YR2_t2.bmp')
    _gt = Image.open('./Dataset/YR2_gt.bmp').convert('1')
    _gt = transforms.ToTensor()(_gt)
    change_pixels = _gt.sum()
    total_pixels = _gt.numel()
    print(change_pixels / total_pixels)
    _t2 = torch.log1p(_t2)
    _t2 = (_t2 - _t2.min()) / (_t2.max() - _t2.min())
    _t1 = torch.log1p(_t1)
    _t1 = (_t1 - _t1.min()) / (_t1.max() - _t1.min())
    return _t2, _t1, _gt

def _YR4():
    _t1 = read_img('./Dataset/YR4_t1.bmp')
    _t2 = read_img('./Dataset/YR4_t2.bmp')
    _gt = Image.open('./Dataset/YR4_gt.bmp').convert('1')
    _gt = transforms.ToTensor()(_gt)
    change_pixels = _gt.sum()
    total_pixels = _gt.numel()
    print(change_pixels / total_pixels)
    _t2 = torch.log1p(_t2)
    _t2 = (_t2 - _t2.min()) / (_t2.max() - _t2.min())
    _t1 = torch.log1p(_t1)
    _t1 = (_t1 - _t1.min()) / (_t1.max() - _t1.min())
    return _t2, _t1, _gt


def _Toulouse():
    _t1 = read_img('./Dataset/Toulouse_t1.png')
    _t2 = read_img('./Dataset/Toulouse_t2.png')
    _gt = Image.open('./Dataset/Toulouse_gt.png').convert('1')
    _gt = transforms.ToTensor()(_gt)
    change_pixels = _gt.sum()
    total_pixels = _gt.numel()
    print(change_pixels / total_pixels)
    scale = 0.2
    _, H, W = _t1.shape
    new_H = int(H * scale)
    new_W = int(W * scale)

    _t1 = F.resize(_t1, [new_H, new_W], interpolation=transforms.InterpolationMode.BILINEAR)
    _t2 = F.resize(_t2, [new_H, new_W], interpolation=transforms.InterpolationMode.BILINEAR)
    _gt = F.resize(_gt, [new_H, new_W], interpolation=transforms.InterpolationMode.NEAREST)
    return _t1, _t2, _gt


DATASETS = {
    "Italy": _Italy,
    "Gloucester": _Gloucester,
    'California': _California,
    'Ottawa': _Ottawa,
    'YR1': _YR1,
    'YR2': _YR2,
    'YR3': _YR3,
    'YR4': _YR4,
    'Texas': _Texas,
    'Toulouse': _Toulouse,
}


if __name__ == '__main__':
    _Texas()

