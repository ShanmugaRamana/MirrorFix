# utils/metrics.py
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(img1, img2):
    img1 = img1.cpu().detach().numpy().transpose(0, 2, 3, 1)
    img2 = img2.cpu().detach().numpy().transpose(0, 2, 3, 1)
    # Clip values to valid range
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    psnr_values = [psnr(img1[i], img2[i], data_range=1) for i in range(img1.shape[0])]
    return np.mean(psnr_values)

def calculate_ssim(img1, img2):
    img1 = img1.cpu().detach().numpy().transpose(0, 2, 3, 1)
    img2 = img2.cpu().detach().numpy().transpose(0, 2, 3, 1)
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    ssim_values = [ssim(img1[i], img2[i], multichannel=True, data_range=1) 
                  for i in range(img1.shape[0])]
    return np.mean(ssim_values)