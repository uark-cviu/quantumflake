import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
import copy
from tqdm import tqdm

from ..models.classifier import FlakeLayerClassifier

def train(
    data_dir,
    epochs=30,
    lr=1e-3,
    batch_size=32,
    val_split=0.2,
    device="cpu",
    save_dir="runs/classify",
    num_materials=2,
    material_dim=64,
    freeze_cnn=False,
):
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

    pin = str(device).startswith("cuda")
    num_workers = 4 if pin else 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    print(f"Training with {len(train_dataset)} images, validating with {len(val_dataset)} images.")

    model = FlakeLayerClassifier(
        num_materials=num_materials,
        material_dim=material_dim,
        num_classes=num_classes,
        freeze_cnn=freeze_cnn,
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
            outputs = model(inputs)  # material=None for image-only training
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
            out_file = save_path / 'best_classifier.pth'
            print(f"  -> New best model found! Saving to {out_file}")
            torch.save({
                'model_state_dict': best_model_state,
                'class_names': class_names,
                'num_classes': num_classes,
                'num_materials': num_materials,
                'material_dim': material_dim,
            }, out_file)

    print("\n--- Classifier Training Finished ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved in: {save_path}")
