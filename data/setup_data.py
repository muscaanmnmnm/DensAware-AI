import kagglehub
import os

print("Downloading dataset... this might take a few minutes...")

# Download latest version
path = kagglehub.dataset_download("shantanughosh/vindr-mammogram-dataset-dicom-to-png")

print("\nSUCCESS! Dataset is stored here:")
print(path)

print("\nChecking for metadata file...")
# Check if the csv exists in that path
files = os.listdir(path)
if 'finding_annotations.csv' in files:
    print("Found 'finding_annotations.csv'. We are good to go!")
else:
    print("Files found:", files)