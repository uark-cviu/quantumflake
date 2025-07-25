import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import copy
from tqdm import tqdm

from ..models.classifier import FlakeLayerClassifier

def train(data_dir, epochs=30, lr=1e-3, batch_size=32, val_split=0.2, device="cpu", save_dir="runs/classify", num_materials=4, material_dim=64):
    """
    Trains the FlakeLayerClassifier model.

    Args:
        data_dir (str): Path to the root of the image dataset.
                        Should contain subdirectories for each class (e.g., '1-layer', '2-layer').
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size for training and validation.
        val_split (float): Fraction of data to use for validation (e.g., 0.2 for 20%).
        device (str): Device to train on ('cpu', 'cuda:0', etc.).
        save_dir (str): Directory to save training runs and the best model.
        num_materials (int): The number of materials for the embedding layer.
        material_dim (int): The dimension of the material embedding vector.
    """
    print("\n Starting Classifier Training")
    data_path = Path(data_dir)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = ImageFolder(data_path)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    indices = list(range(len(full_dataset)))
    labels = [s[1] for s in full_dataset.samples]
    train_indices, val_indices = train_test_split(indices, test_size=val_split, stratify=labels, random_state=42)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    train_dataset.dataset = copy.copy(full_dataset)
    train_dataset.dataset.transform = train_tf
    val_dataset.dataset.transform = val_tf

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Training with {len(train_dataset)} images, validating with {len(val_dataset)} images.")

    model = FlakeLayerClassifier(
        num_materials=num_materials, 
        material_dim=material_dim, 
        num_classes=num_classes
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) # material=None for image-only training
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        val_acc = val_correct / len(val_dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best model found! Saving to {save_path / 'best_classifier.pth'}")
            torch.save({
                'model_state_dict': best_model_state,
                'class_names': class_names,
                'num_classes': num_classes,
                'num_materials': num_materials,
                'material_dim': material_dim,
            }, save_path / 'best_classifier.pth')

    print("\n--- Classifier Training Finished ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved in: {save_path}")
