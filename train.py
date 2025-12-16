import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import your custom files
from dataset import VinDrMammoDataset
from model import DensityAwareModel

# --- CONFIGURATION ---
CSV_PATH = "data/finding_annotations.csv"
IMG_ROOT = r"C:\Users\Marhaba\.cache\kagglehub\datasets\shantanughosh\vindr-mammogram-dataset-dicom-to-png\versions\1\images_png"

BATCH_SIZE = 16
LEARNING_RATE = 0.0001   # Lowered LR for stability
NUM_EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, clinical, labels in loader:
            images, clinical, labels = images.to(DEVICE), clinical.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images, clinical)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()
            
            # --- FIX: Apply Sigmoid manually here since we removed it from model ---
            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    
    # Print a mini-report during validation to see if we are finding cancer
    print("\n   [Validation Report]")
    # We use zero_division=0 to silence the warning if we still predict 0 cancers
    print(classification_report(all_labels, all_preds, target_names=["Healthy", "Abnormal"], zero_division=0))
    
    return avg_loss, acc

def train():
    print(f"--- Starting Training on {DEVICE} ---")
    
    # 1. PREPARE DATA
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading Data...")
    full_dataset = VinDrMammoDataset(CSV_PATH, IMG_ROOT, transform=transform)
    
    # Use 1000 images
    subset_size = 1000 
    subset_dataset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])
    
    train_size = int(0.8 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    train_data, val_data = random_split(subset_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # 2. SETUP MODEL & WEIGHTED LOSS
    model = DensityAwareModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- CRITICAL ENGINEERING FIX ---
    # We calculate that Healthy cases are ~15x more common than Cancer.
    # So we give Cancer cases a weight of 15.0 to balance the scales.
    pos_weight = torch.tensor([15.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 3. TRAINING LOOP
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        for batch_idx, (images, clinical, labels) in enumerate(train_loader):
            images, clinical, labels = images.to(DEVICE), clinical.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images, clinical)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"   >> RESULT: Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")

    # 4. SAVE
    torch.save(model.state_dict(), "breast_cancer_model_v2.pth")
    print("\n✅ Re-Training Complete! Model Updated.")

if __name__ == "__main__":
    train()