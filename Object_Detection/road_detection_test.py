import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 2
num_steps = 8
beta = 0.9
num_classes = 2
image_size = (128, 128)
epochs = 10
lr = 0.001

# Spiking Model inspired by SAD architecture
class SpikingSAD(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(64)
        self.bottleneck_lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        # Decoder
        self.dec_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.dec_conv2 = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Initialize potentials
        mem_enc_lif1 = self.enc_lif1.init_leaky()
        mem_enc_lif2 = self.enc_lif2.init_leaky()
        mem_bottleneck_lif = self.bottleneck_lif.init_leaky()
        mem_dec_lif1 = self.dec_lif1.init_leaky()
        
        # Pre-calculate output size
        B, _, H, W = x.shape
        outputs = torch.zeros((num_steps, B, num_classes, H, W), device=device)
        
        for step in range(num_steps):
            # Encoder
            x1 = self.enc_conv1(x)
            x1 = self.enc_bn1(x1)
            spk1, mem_enc_lif1 = self.enc_lif1(x1, mem_enc_lif1)
            
            x2 = self.enc_conv2(spk1)
            x2 = self.enc_bn2(x2)
            spk2, mem_enc_lif2 = self.enc_lif2(x2, mem_enc_lif2)
            
            # Bottleneck
            x3 = self.bottleneck_conv(spk2)
            x3 = self.bottleneck_bn(x3)
            spk3, mem_bottleneck_lif = self.bottleneck_lif(x3, mem_bottleneck_lif)
            
            # Decoder
            x4 = self.dec_upsample(spk3)
            x4 = self.dec_conv1(x4)
            x4 = self.dec_bn1(x4)
            spk4, mem_dec_lif1 = self.dec_lif1(x4, mem_dec_lif1)
            
            out = self.dec_conv2(spk4)
            outputs[step] = out
        
        # Average outputs over time steps
        return outputs.mean(0)

# Custom dataset with corrected mask
class SplitDataset(Dataset):
    def __init__(self, root_dir, image_paths, image_size=(128, 128)):
        self.root_dir = root_dir
        self.image_paths = image_paths
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Transform image
        img_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = img_transform(img)
        
        # Get corresponding mask path
        filename = os.path.basename(img_path)
        
        # Handle different naming patterns based on prefix
        if filename.startswith('um_'):
            mask_name = filename.replace('um_', 'um_lane_')
        elif filename.startswith('umm_'):
            mask_name = filename.replace('umm_', 'umm_road_')
        elif filename.startswith('uu_'):
            mask_name = filename.replace('uu_', 'uu_road_')
        else:
            mask_name = filename.replace('.png', '_road.png')
        
        mask_path = os.path.join(self.root_dir, 'training', 'gt_image_2', mask_name)
        
        # If the first pattern doesn't exist, try an alternative to find the corresponding mask
        if not os.path.exists(mask_path):
            alt_mask_name = filename.replace('.png', '_road.png')
            alt_mask_path = os.path.join(self.root_dir, 'training', 'gt_image_2', alt_mask_name)
            if os.path.exists(alt_mask_path):
                mask_path = alt_mask_path
            else:
                raise FileNotFoundError(f"Mask not found for {img_path}")
        
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)
        
        # Turn multiclassification mask to a binary mask: road = 1, non-road = 0
        mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)
        mask[(mask_array[:, :, 0] == 255) & (mask_array[:, :, 1] == 0) & (mask_array[:, :, 2] == 0)] = 1
        
        mask = Image.fromarray(mask)
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        
        return img_tensor, mask_tensor

# Create datasets
root_dir = 'data_road' # path to the dataset

# Get all training image paths
all_image_paths = sorted(glob.glob(os.path.join(root_dir, 'training', 'image_2', '*.png')))
train_paths, val_paths = train_test_split(all_image_paths, test_size=0.2, random_state=42)

# Create datasets using the custom split
train_dataset = SplitDataset(root_dir, train_paths, image_size=image_size)
val_dataset = SplitDataset(root_dir, val_paths, image_size=image_size)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize model
model = SpikingSAD().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training history
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == masks).sum().item()
        total += masks.numel()
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    
    # Validation during training phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == masks).sum().item()
            total += masks.numel()
    
    val_loss /= len(val_loader)
    val_acc = correct / total
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f'Epoch {epoch+1}/{epochs}: '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Test on Sample Images
def visualize_results(model, data_loader, num_samples=3):
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(data_loader):
            if i >= num_samples:
                break
                
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            # Denormalize images
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            denorm_images = images * std + mean
            denorm_images = denorm_images.clamp(0, 1)
            
            # Plot results
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(denorm_images[0].permute(1, 2, 0))
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(masks[0], cmap='jet', vmin=0, vmax=1)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(predictions[0], cmap='jet', vmin=0, vmax=1)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'test_result_{i}.png')
            plt.show()

# Visualize validation results
visualize_results(model, val_loader)