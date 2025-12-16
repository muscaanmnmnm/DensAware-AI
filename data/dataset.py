import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VinDrMammoDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # MAPPINGS: Convert text to numbers
        # 1. Density: A=0, B=1, C=2, D=3
        self.density_map = {'DENSITY A': 0, 'DENSITY B': 1, 'DENSITY C': 2, 'DENSITY D': 3}
        
        # 2. View Position: CC=0, MLO=1
        self.view_map = {'CC': 0, 'MLO': 1}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # --- A. GET IMAGE PATH ---
        # The folder structure in VinDr is usually: root/study_id/image_id.png
        row = self.annotations.iloc[idx]
        study_id = str(row['study_id'])
        image_id = str(row['image_id'])
        
        # Construct path
        img_path = os.path.join(self.root_dir, study_id, image_id + '.png')
        
        # --- B. LOAD IMAGE ---
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Fallback for safety (create a black image if file missing)
            # This prevents crashing during long training
            print(f"Warning: Missing image {img_path}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        # --- C. GET CLINICAL DATA (TABULAR) ---
        # We use Density and View Position
        density_val = self.density_map.get(row['breast_density'], 0) # Default to 0 if missing
        view_val = self.view_map.get(row['view_position'], 0)
        
        # Create a vector: [density, view_position]
        # In a real paper, we might One-Hot Encode density, but this is fine for now.
        clinical_features = torch.tensor([density_val, view_val], dtype=torch.float32)

        # --- D. GET LABEL (TARGET) ---
        # Logic: If BI-RADS is 1 or 2 -> Healthy (0). If 3, 4, 5 -> Abnormal (1).
        birads = row['breast_birads']
        
        # Handle cases where birads might be "BI-RADS 1" string or just number 1
        try:
            birads_score = int(str(birads).split()[-1]) # Extract number
        except:
            birads_score = 1 # Default to healthy if unclear

        label = 1 if birads_score > 2 else 0
        
        return image, clinical_features, torch.tensor(label, dtype=torch.float32)

# --- TEST BLOCK (Run this file directly to check) ---
if __name__ == "__main__":
    # Define paths (Update these if yours are different!)
    # Note: Using the paths you confirmed in Step 1
    CSV_PATH = "data/finding_annotations.csv" 
    IMG_ROOT = r"C:\Users\Marhaba\.cache\kagglehub\datasets\shantanughosh\vindr-mammogram-dataset-dicom-to-png\versions\1\images_png"

    print("Testing Dataset Class...")
    
    # Simple transform for testing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Initialize
    dataset = VinDrMammoDataset(csv_file=CSV_PATH, root_dir=IMG_ROOT, transform=test_transform)
    
    # Get one sample
    print(f"Dataset Length: {len(dataset)}")
    img, clinic, lbl = dataset[0]
    
    print("\n--- Sample 0 ---")
    print(f"Image Shape: {img.shape} (Should be 3, 224, 224)")
    print(f"Clinical Data: {clinic} (Density, View)")
    print(f"Label: {lbl} (0=Healthy, 1=Abnormal)")
    print("\n✅ Dataset Class is working!")