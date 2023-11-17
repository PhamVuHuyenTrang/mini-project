import torch
import torchvision.transforms.functional as F

def normalize(x, mean, std):
    assert len(x.shape) == 4
    return (x - torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)) \
        / torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)

def rotate_30_degrees(x, mean, std):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    elif len(x.shape) != 4:
        raise ValueError("Input tensor must have 3 or 4 dimensions (batch dimension)")
    
    rotated_tensors = []
    for img in x:
        img_pil = F.to_pil_image(img)
        rotated_img = F.rotate(img_pil, 30)
        rotated_tensor = F.to_tensor(rotated_img)
        rotated_tensors.append(rotated_tensor)

    rotated_batch = torch.stack(rotated_tensors, dim=0)

    normalized_batch = normalize(rotated_batch, mean, std)
    return normalized_batch.squeeze(0) if len(x.shape) == 3 else normalized_batch

def rotate_60_degrees(x, mean, std):
    if len(x.shape) != 4:
        raise ValueError("Input tensor must have 4 dimensions (batch dimension, channels, height, width)")

    rotated_tensors = []
    for img in x:
        img_pil = F.to_pil_image(img)
        rotated_img = F.rotate(img_pil, 60)
        rotated_tensor = F.to_tensor(rotated_img)
        rotated_tensors.append(rotated_tensor)

    rotated_batch = torch.stack(rotated_tensors, dim=0)

    normalized_batch = normalize(rotated_batch, mean, std)
    return normalized_batch

def add_noise(x, mean, std, noise_factor=0.1):
    if len(x.shape) != 4:
        raise ValueError("Input tensor must have 4 dimensions (batch dimension, channels, height, width)")

    noise = torch.randn_like(x) * noise_factor
    noisy_img = x + noise
    noisy_img = torch.clamp(noisy_img, 0, 1)

    normalized_tensor = normalize(noisy_img, mean, std)
    return normalized_tensor

def change_colors(x, mean, std):
    if len(x.shape) != 4:
        raise ValueError("Input tensor must have 4 dimensions (batch dimension, channels, height, width)")

    colored_tensors = []
    for img in x:
        img_pil = F.to_pil_image(img)
        colored_img = F.adjust_saturation(img_pil, 2.0)
        colored_tensor = F.to_tensor(colored_img)
        colored_tensors.append(colored_tensor)

    colored_batch = torch.stack(colored_tensors, dim=0)

    normalized_batch = normalize(colored_batch, mean, std)
    return normalized_batch
