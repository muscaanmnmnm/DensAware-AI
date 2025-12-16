import os
import torch
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

# Import your custom model structure
# Ensure model.py is in the same folder
from model import DensityAwareModel

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MODEL_PATH = "breast_cancer_model_v2.pth"
DEVICE = "cpu" 

# --- LOAD MODEL ---
print("Loading DensAware AI Model...")
model = DensityAwareModel()
try:
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    else:
        print(f"⚠️ Warning: '{MODEL_PATH}' not found. Using random weights for demo.")
    model.eval()
    print("✅ Model Active")
except Exception as e:
    print(f"❌ Model Load Error: {e}")

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DENSITY_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
VIEW_MAP = {'CC': 0, 'MLO': 1}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Handle Image
        if 'file' not in request.files: return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '': return jsonify({'error': 'No selected file'})

        try:
            image = Image.open(file).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)
        except:
            return jsonify({'error': 'Invalid Image File'})

        # 2. Handle Clinical Data
        d_val = DENSITY_MAP.get(request.form.get('density'), 0)
        v_val = VIEW_MAP.get(request.form.get('view'), 0)
        clinical_tensor = torch.tensor([[float(d_val), float(v_val)]])

        # 3. Inference
        with torch.no_grad():
            output = model(image_tensor, clinical_tensor)
            probability = torch.sigmoid(output).item()

        # 4. Result
        result = "Abnormal (High Risk)" if probability > 0.5 else "Normal (Low Risk)"
        
        return jsonify({
            'prediction': result,
            'confidence': f"{probability * 100:.2f}%",
            'risk_score': probability
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)