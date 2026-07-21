import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import platform

# Automatically drops to 0 on Mac
NUM_WORKERS = 0 if platform.system() == "Darwin" else 2

class GaussianBlur:
    def __init__(self, kernel_size=5, sigma=2.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
    def __call__(self, img):
        import torchvision.transforms.functional as F
        return F.gaussian_blur(img, [self.kernel_size, self.kernel_size], [self.sigma, self.sigma])

class AddGaussianNoise:
    def __init__(self, std=0.15):
        self.std = std
    def __call__(self, tensor):
        return tensor + self.std * torch.randn_like(tensor)


class BrightnessContrast:
    def __init__(self, brightness_factor=1.6, contrast_factor=1.6):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
    def __call__(self, img):
        import torchvision.transforms.functional as F
        img = F.adjust_brightness(img, self.brightness_factor)
        img = F.adjust_contrast(img, self.contrast_factor)
        return torch.clamp(img, 0.0, 1.0)


class Rotation:
    """Rotates the image by a fixed angle (simulates sensor/orientation shift)."""
    def __init__(self, degrees=30):
        self.degrees = degrees
    def __call__(self, img):
        import torchvision.transforms.functional as F
        return F.rotate(img, self.degrees)


class Occlusion:
    """Cutout-style occlusion: zeroes out a random square patch of the image."""
    def __init__(self, patch_size=12):
        self.patch_size = patch_size
    def __call__(self, tensor):
        _, h, w = tensor.shape
        ps = self.patch_size
        top = torch.randint(0, h - ps + 1, (1,)).item()
        left = torch.randint(0, w - ps + 1, (1,)).item()
        tensor = tensor.clone()
        tensor[:, top:top + ps, left:left + ps] = 0.0
        return tensor


class JPEGCompression:
    """Applies lossy JPEG re-encoding to simulate compression-artifact shift."""
    def __init__(self, quality=15):
        self.quality = quality
    def __call__(self, img):
        import io
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        from PIL import Image
        return Image.open(buffer).convert("RGB")


def get_clean_dataloaders(data_dir="data/raw", batch_size=128):
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(), norm
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), norm])
    
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS),
            DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS))


def get_shifted_dataloader(shift_type="blur", severity=1, data_dir="data/raw", batch_size=128):
    """
    Returns a DataLoader for a specific distribution shift and severity level (1..5).
    """
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    # Define severity maps for each shift type (levels 1 through 5)
    if shift_type == "blur":
        # Kernel size must remain odd: [3, 5, 7, 9, 11]
        kernel_sizes = [3, 5, 7, 9, 11]
        sigmas = [0.5, 1.5, 2.5, 3.5, 4.5]
        k = kernel_sizes[severity - 1]
        sig = sigmas[severity - 1]
        tfs = transforms.Compose([transforms.ToTensor(), GaussianBlur(kernel_size=k, sigma=sig), norm])

    elif shift_type == "noise":
        stds = [0.05, 0.10, 0.15, 0.20, 0.25]
        tfs = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(std=stds[severity - 1]), norm])

    elif shift_type == "brightness_contrast":
        factors = [1.2, 1.4, 1.6, 1.8, 2.0]
        f = factors[severity - 1]
        tfs = transforms.Compose([transforms.ToTensor(), BrightnessContrast(brightness_factor=f, contrast_factor=f), norm])

    elif shift_type == "rotation":
        degrees = [15, 30, 45, 60, 75]
        tfs = transforms.Compose([transforms.ToTensor(), Rotation(degrees=degrees[severity - 1]), norm])

    elif shift_type == "occlusion":
        patch_sizes = [4, 8, 12, 16, 20]
        tfs = transforms.Compose([transforms.ToTensor(), Occlusion(patch_size=patch_sizes[severity - 1]), norm])

    elif shift_type == "jpeg":
        # Lower quality value = higher compression artifact severity
        qualities = [50, 35, 20, 10, 5]
        tfs = transforms.Compose([JPEGCompression(quality=qualities[severity - 1]), transforms.ToTensor(), norm])

    else:
        raise ValueError(f"Unknown shift: {shift_type}")
        
    shifted_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfs)
    
    return DataLoader(shifted_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
