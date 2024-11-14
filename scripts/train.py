# scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path to import from other directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.transformations import ReflectionTransforms
from scripts.dataset_loader import ReflectionDataset

class DeepReflectionRemoval(nn.Module):
    def __init__(self):
        super(DeepReflectionRemoval, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            self._make_layer(3, 64),
            nn.MaxPool2d(2),
            self._make_layer(64, 128),
            nn.MaxPool2d(2),
            self._make_layer(128, 256),
            nn.MaxPool2d(2),
        )
        
        # Bridge
        self.bridge = self._make_layer(256, 512)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            self._make_layer(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            self._make_layer(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            self._make_layer(64, 64),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh()
        )
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.bridge(x)
        x = self.decoder(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=5, 
                                                   verbose=True)
    
    best_val_loss = float('inf')
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            reconstruction_loss = criterion(outputs, targets)
            perceptual_loss = torch.mean(torch.abs(
                torch.mean(outputs, dim=1) - torch.mean(targets, dim=1)))
            
            total_loss = reconstruction_loss + 0.1 * perceptual_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join('models', 'reflection_removal_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Model saved with validation loss: {val_loss:.4f}')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize transforms
    transforms = ReflectionTransforms()
    
    # Create datasets
    train_dataset = ReflectionDataset(
        'dataset/train/input',
        'dataset/train/target',
        transform=transforms.train_transforms
    )
    
    val_dataset = ReflectionDataset(
        'dataset/test/input',
        'dataset/test/target',
        transform=transforms.test_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, 
                          shuffle=False, num_workers=0)
    
    # Initialize model
    model = DeepReflectionRemoval().to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=100, device=device)

if __name__ == '__main__':
    main()