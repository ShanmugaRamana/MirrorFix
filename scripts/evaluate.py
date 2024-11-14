# scripts/evaluate.py
import torch
from utils.metrics import calculate_psnr, calculate_ssim
import torchvision.utils as vutils
import os

def evaluate_model(model, test_loader, device='cuda', save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Denormalize images
            def denorm(tensor):
                return tensor * 0.5 + 0.5
            
            # Calculate metrics
            psnr = calculate_psnr(denorm(outputs), denorm(targets))
            ssim = calculate_ssim(denorm(outputs), denorm(targets))
            
            total_psnr += psnr
            total_ssim += ssim
            
            # Save results
            if i < 5:  # Save first 5 results
                vutils.save_image(denorm(outputs), 
                                os.path.join(save_dir, f'output_{i}.png'))
                vutils.save_image(denorm(inputs), 
                                os.path.join(save_dir, f'input_{i}.png'))
                vutils.save_image(denorm(targets), 
                                os.path.join(save_dir, f'target_{i}.png'))
    
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    
    print(f'Evaluation Results:')
    print(f'Average PSNR: {avg_psnr:.2f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')