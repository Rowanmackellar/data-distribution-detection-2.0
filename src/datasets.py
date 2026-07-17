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
    """Pushes brightness and contrast away from the training distribution."""
    def __init__(self, brightness_factor=1.6, contrast_factor=1.6):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
    def __call__(self, img):
        import torchvision.transforms.functional as F
        img = F.adjust_brightness(img, self.brightness_factor)
        img = F.adjust_contrast(img, self.contrast_factor)
        return img


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


def get_shifted_dataloader(shift_type="blur", data_dir="data/raw", batch_size=128):
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    if shift_type == "blur":
        tfs = transforms.Compose([transforms.ToTensor(), GaussianBlur(), norm])
    elif shift_type == "noise":
        tfs = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(), norm])
    elif shift_type == "brightness_contrast":
        tfs = transforms.Compose([transforms.ToTensor(), BrightnessContrast(), norm])
    elif shift_type == "rotation":
        tfs = transforms.Compose([transforms.ToTensor(), Rotation(), norm])
    elif shift_type == "occlusion":
        tfs = transforms.Compose([transforms.ToTensor(), Occlusion(), norm])
    elif shift_type == "jpeg":
        # JPEGCompression needs a PIL image, so it runs before ToTensor()
        tfs = transforms.Compose([JPEGCompression(), transforms.ToTensor(), norm])
    else:
        raise ValueError(f"Unknown shift: {shift_type}")
        
    shifted_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfs)
    
    return DataLoader(shifted_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)