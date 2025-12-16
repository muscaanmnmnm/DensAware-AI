import pandas as pd
import os

# 1. Setup Paths
# Use the path you found in Step 1
image_path = r"C:\Users\Marhaba\.cache\kagglehub\datasets\shantanughosh\vindr-mammogram-dataset-dicom-to-png\versions\1\images_png"
csv_path = "data/finding_annotations.csv"

# 2. Load the CSV
print(f"Loading labels from {csv_path}...")
try:
    df = pd.read_csv(csv_path)
    print("✅ CSV Loaded Successfully!")
    print(f"Total Rows: {len(df)}")
    print("\nColumns found:", df.columns.tolist())
    
    # 3. Check for Clinical Features (Critical for your project)
    if 'breast_density' in df.columns:
        print("\n✅ Found 'breast_density' - We can use this for the Fusion Model!")
        print("Density Categories:", df['breast_density'].unique())
    else:
        print("\n❌ Warning: 'breast_density' column missing.")

    # 4. Check if Image IDs match the Folder
    print("\nChecking if CSV matches Image Folder...")
    folder_images = os.listdir(image_path)
    # The folder might contain subfolders (patient IDs) or just images. Let's check.
    first_item = folder_images[0]
    print(f"First item in image folder: {first_item}")
    
    # We will do a deeper match in Step 3, this is just a quick check.
    
except FileNotFoundError:
    print("❌ Error: Could not find 'data/finding_annotations.csv'. Did you move the file there?")