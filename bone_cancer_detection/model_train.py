#model Training
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from efficientnet_pytorch import EfficientNet

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformations
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Load dataset and split into training and validation
def load_data(dataset_path, transform, batch_size=32):
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

# Load and modify the EfficientNet B4 model
def initialize_model(num_classes=3):
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model.to(device)

# Training the model
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_accuracy = 100 * train_correct / train_total
    return running_loss / len(train_loader), train_accuracy

# Validate the model
def validate(model, val_loader):
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    return 100 * val_correct / val_total

# Main training loop
def main():
    dataset_path = r'C:\Users\pytorch\Desktop\Machine-Learning-Projects\bone cancer detection\dataset_augmented'
    transform = get_transforms()
    train_loader, val_loader = load_data(dataset_path, transform)
    
    model = initialize_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    num_epochs = 50
    best_accuracy = 0.0
    patience = 5
    no_improve_count = 0

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        
        val_accuracy = validate(model, val_loader)
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_bone_cancer_model_efficientnet.pth')
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print("Early stopping triggered")
            break
        
        scheduler.step()
    
    # Load the best model for final evaluation
    model.load_state_dict(torch.load('best_bone_cancer_model_efficientnet.pth'))
    final_val_accuracy = validate(model, val_loader)
    print(f'Final Validation Accuracy: {final_val_accuracy:.2f}%')

if __name__ == '__main__':
    main()
