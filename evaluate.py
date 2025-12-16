import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Import your custom files
from dataset import VinDrMammoDataset
from model import DensityAwareModel

# --- CONFIGURATION ---
CSV_PATH = "data/finding_annotations.csv"
IMG_ROOT = r"C:\Users\Marhaba\.cache\kagglehub\datasets\shantanughosh\vindr-mammogram-dataset-dicom-to-png\versions\1\images_png"
MODEL_PATH = "breast_cancer_model_v2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate():
    print(f"--- Loading Model from {MODEL_PATH} ---")
    
    # 1. LOAD DATA (Same way as training)
    # We use a transform to normalize the images exactly like training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = VinDrMammoDataset(CSV_PATH, IMG_ROOT, transform=transform)
    
    # Use the same subset logic (or use a fresh 200 images for pure testing)
    # Here we take the LAST 200 images to ensure they are likely different from the training set
    test_size = 200
    _, test_dataset = random_split(full_dataset, [len(full_dataset) - test_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 2. LOAD MODEL
    model = DensityAwareModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Set to evaluation mode
    
    print(f"Testing on {len(test_dataset)} images...")

    # 3. RUN PREDICTIONS
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, clinical, labels in test_loader:
            images, clinical = images.to(DEVICE), clinical.to(DEVICE)
            
            outputs = model(images, clinical)
            
            # Convert probability to 0 or 1
            preds = (outputs.squeeze() > 0.5).float()
            
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    # 4. GENERATE REPORT
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=["Healthy (0)", "Abnormal (1)"]))
    
    # 5. PLOT CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Healthy", "Abnormal"], yticklabels=["Healthy", "Abnormal"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Density Aware Model')
    
    # Save the plot
    plt.savefig('confusion_matrix.png')
    print("\n✅ Graph saved as 'confusion_matrix.png'")
    print("Check your folder to see the image!")

if __name__ == "__main__":
    evaluate()