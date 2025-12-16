import torch
import torch.nn as nn
import torchvision.models as models

class DensityAwareModel(nn.Module):
    def __init__(self):
        super(DensityAwareModel, self).__init__()
        
        # --- BRANCH 1: IMAGE PROCESSING (The "Eye") ---
        # We use ResNet18, a standard powerful CNN
        # pretrained=True means it already knows how to see basic shapes
        self.cnn = models.resnet18(pretrained=True)
        
        # Remove the last layer of ResNet because we don't want it to classify yet.
        # We just want the "features" (the understanding of the image).
        # ResNet18 outputs 512 features before the final layer.
        self.cnn.fc = nn.Identity() 
        
        # --- BRANCH 2: CLINICAL DATA (The "Context") ---
        # Input: 2 numbers (Density, View)
        # Output: 16 numbers (features)
        self.clinical_net = nn.Sequential(
            nn.Linear(2, 16),      # Take 2 inputs, expand to 16
            nn.ReLU(),             # Activation function
            nn.Linear(16, 32),     # Expand to 32
            nn.ReLU()
        )
        
        # --- FUSION LAYER (Combining both) ---
        # 512 (from Image) + 32 (from Clinical) = 544 Total Inputs
        self.fusion = nn.Sequential(
            nn.Linear(512 + 32, 128),  # Compress to 128
            nn.ReLU(),
            nn.Dropout(0.3),           # Drop 30% neurons to prevent memorization
            nn.Linear(128, 1)          # Final Output: 1 number (Cancer Score)
        )
        
        # The Sigmoid function squashes the output between 0 and 1 (Probability)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, clinical_data):
        # 1. Pass image through CNN
        image_features = self.cnn(image) 
        
        # 2. Pass clinical data through small network
        clinical_features = self.clinical_net(clinical_data)
        
        # 3. Concatenate (Fuse) them together
        combined = torch.cat((image_features, clinical_features), dim=1) 
        
        # 4. Final Prediction
        output = self.fusion(combined)
        
        # --- CRITICAL CHANGE FOR STEP 6 ---
        # We REMOVED self.sigmoid(output)
        # We return the raw output (logits) so the Loss function can handle the math.
        return output

# --- TEST BLOCK ---
if __name__ == "__main__":
    print("Initializing the Dual-Stream Model...")
    # This might take 10-20 seconds to download ResNet18 the first time
    model = DensityAwareModel()
    
    # Create fake data to test
    fake_image = torch.randn(2, 3, 224, 224) # 2 images (Batch size 2)
    fake_clinical = torch.randn(2, 2)        # 2 clinical records
    
    # Run the model
    prediction = model(fake_image, fake_clinical)
    
    print("\n--- Model Architecture Test ---")
    print(f"Input Image Shape: {fake_image.shape}")
    print(f"Input Clinical Shape: {fake_clinical.shape}")
    print(f"Output Shape: {prediction.shape} (Should be [2, 1])")
    print(f"Prediction Values:\n{prediction}")
    print("\n✅ Model Architecture is ready for training!")