import torch
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
from scripts.dataset_loader import ReflectionDataset
from utils.transformations import ReflectionTransforms
from torch.utils.data import DataLoader
from scripts.train import DeepReflectionRemoval, train_model
from scripts.evaluate import evaluate_model
import matplotlib.pyplot as plt

def process_single_image(model, image_path, output_path, device='cuda'):
    # Process image logic remains unchanged
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        # Denormalize and convert output
        output = output.cpu().squeeze(0)
        output = output * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                 torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        output = output.clamp(0, 1)
        
        output_image = transforms.ToPILImage()(output)
        output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
        
        output_image.save(output_path)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(output_image)
        plt.title('Reflection Removed')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Processed image saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Reflection Removal')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'process'],
                        help='Mode: train, evaluate, or process single image')
    parser.add_argument('--data_dir', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--results_dir', default='results', help='Directory to save results')
    parser.add_argument('--input_image', help='Path to input image for processing')
    parser.add_argument('--output_image', help='Path to save processed image')
    
    # Get args; skip errors if args are missing for interactive mode
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(mode=None)

    # Interactive mode: ask for input if mode is not provided
    if not args.mode:
        args.mode = input("Choose mode (train, evaluate, process): ").strip()
    if args.mode == 'process':
        if not args.input_image:
            args.input_image = input("Enter the path to the input image: ").strip()
        if not args.output_image:
            output_dir = 'processed_images'
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.splitext(os.path.basename(args.input_image))[0]
            args.output_image = os.path.join(output_dir, f"{filename}_no_reflection.png")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = DeepReflectionRemoval().to(device)

    if args.mode == 'process':
        model_path = 'models/reflection_removal_model.pth'
        if not os.path.exists(model_path):
            print("Error: Trained model not found. Please train the model first.")
            return

        model.load_state_dict(torch.load(model_path))
        success = process_single_image(model, args.input_image, args.output_image, device)
        if success:
            print("Image processing completed successfully!")
        else:
            print("Image processing failed!")
    elif args.mode == 'train':
        if not args.data_dir:
            args.data_dir = input("Enter path to dataset directory: ").strip()

        transforms = ReflectionTransforms()
        train_dataset = ReflectionDataset(
            os.path.join(args.data_dir, 'train/input'),
            os.path.join(args.data_dir, 'train/target'),
            transform=transforms.train_transforms
        )
        
        val_dataset = ReflectionDataset(
            os.path.join(args.data_dir, 'test/input'),
            os.path.join(args.data_dir, 'test/target'),
            transform=transforms.test_transforms
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        train_model(model, train_loader, val_loader, num_epochs=args.num_epochs, device=device)
    else:
        if not args.data_dir:
            args.data_dir = input("Enter path to dataset directory: ").strip()

        model.load_state_dict(torch.load('models/reflection_removal_model.pth'))
        
        test_dataset = ReflectionDataset(
            os.path.join(args.data_dir, 'test/input'),
            os.path.join(args.data_dir, 'test/target'),
            transform=transforms.test_transforms
        )
        
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        evaluate_model(model, test_loader, device=device, save_dir=args.results_dir)

if __name__ == '__main__':
    main()
