import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np

# Import your custom files
from dataset import VinDrMammoDataset
from model import DensityAwareModel

# --- CONFIGURATION ---
CSV_PATH = "data/finding_annotations.csv"
IMG_ROOT = r"C:\Users\Marhaba\.cache\kagglehub\datasets\shantanughosh\vindr-mammogram-dataset-dicom-to-png\versions\1\images_png"
MODEL_PATH = "breast_cancer_model_v2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_visuals():
    print("--- Generating Research Plots ---")
    
    # 1. LOAD DATA (Tabular only for stats)
    df = pd.read_csv(CSV_PATH)
    
    # CLEAN DATA (Just like in dataset.py logic)
    # Map BIRADS to Cancer (1) or Healthy (0)
    def get_label(birads):
        try:
            score = int(str(birads).split()[-1])
        except:
            score = 1
        return 1 if score > 2 else 0

    df['target'] = df['breast_birads'].apply(get_label)
    
    # -------------------------------------------------------
    # PLOT 1: Diagnosis Distribution (Replaces "A Count of Diagnoses")
    # -------------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='pastel')
    plt.title('Distribution of Diagnoses (0=Healthy, 1=Abnormal)')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.savefig('Images/distribution_diagnosis.png')
    print("✅ Saved 'Images/distribution_diagnosis.png'")

    # -------------------------------------------------------
    # PLOT 2: Cancer Rate by Density (Replaces "Heatmap")
    # -------------------------------------------------------
    # This proves your "Density-Aware" hypothesis
    plt.figure(figsize=(8, 5))
    
    # Calculate percentage of cancer in each density group
    density_stats = df.groupby('breast_density')['target'].mean().reset_index()
    density_stats['target'] = density_stats['target'] * 100 # Convert to %
    
    sns.barplot(x='breast_density', y='target', data=density_stats, palette='Reds')
    plt.title('Cancer Rate per Breast Density Category')
    plt.ylabel('Percentage of Abnormal Cases (%)')
    plt.xlabel('Density Category')
    plt.savefig('Images/density_analysis.png')
    print("✅ Saved 'Images/density_analysis.png'")

    # -------------------------------------------------------
    # PLOT 3: Model Predictions on Real Images (NEW & IMPRESSIVE)
    # -------------------------------------------------------
    print("Generating Sample Predictions (This takes a moment)...")
    
    # Setup Model & Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VinDrMammoDataset(CSV_PATH, IMG_ROOT, transform=transform)
    # Get a random subset
    subset, _ = random_split(dataset, [20, len(dataset)-20])
    loader = DataLoader(subset, batch_size=6, shuffle=True)
    
    model = DensityAwareModel().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except:
        print("Warning: Could not load trained model weights. Using random weights for demo.")
    
    model.eval()
    
    # Get one batch
    images, clinical, labels = next(iter(loader))
    images, clinical = images.to(DEVICE), clinical.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(images, clinical)
        probs = torch.sigmoid(outputs.squeeze()) # Convert logits to prob
        preds = (probs > 0.5).float()
    
    # Plot 6 images
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    images = images.cpu()
    
    for i, ax in enumerate(axes.flat):
        if i >= len(images): break
        
        # Un-normalize image for display
        img = images[i].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        ax.imshow(img)
        
        # Title: True vs Pred
        true_lbl = "Cancer" if labels[i]==1 else "Healthy"
        pred_lbl = "Cancer" if preds[i]==1 else "Healthy"
        
        color = 'green' if true_lbl == pred_lbl else 'red'
        
        ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", color=color, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('Images/sample_predictions.png')
    print("✅ Saved 'Images/sample_predictions.png'")

if __name__ == "__main__":
    generate_visuals()