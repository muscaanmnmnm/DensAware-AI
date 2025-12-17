Density-Aware Mammography: A Dual-Stream Multimodal Fusion Framework

📌 Project Overview

This project presents a novel Multimodal Deep Learning Architecture designed to enhance breast cancer detection in mammography. Unlike traditional Computer-Aided Diagnosis (CAD) systems that rely solely on visual data, this framework implements a Dual-Stream Network that fuses:

Visual Stream: High-resolution Mammogram X-rays (processed via a ResNet18 backbone).

Clinical Stream: Patient metadata, specifically Breast Density (BI-RADS) and View Position.

By integrating these two data sources via Late Fusion, the model addresses the critical challenge of tumor occlusion in dense breast tissue, significantly improving sensitivity (Recall) where standard models often fail.

🚀 Key Engineering Features

Dual-Stream Architecture: A custom PyTorch model that processes images and tabular data in parallel before fusing them for the final classification.

Density-Awareness: Explicitly models the relationship between breast density and cancer risk, mitigating the "masking effect" of dense tissue.

Class Imbalance Handling: Implements Weighted BCE With Logits Loss to penalize missed diagnoses 15x more than false alarms, prioritizing high sensitivity.

Real-World Data: Trained and validated on the VinDr-Mammo Dataset (20,000+ images), a large-scale, real-world dataset from Vietnam.

📊 Performance Metrics

Metric

Baseline (Image Only)

Proposed (Density-Aware Fusion)

Recall (Sensitivity)

~0.00 (Failed on Imbalance)

0.54 (Detects Majority of Cancers)

Accuracy

92% (Misleadingly High)

51% (Clinically Valid - High Recall)

Note: The shift in accuracy reflects a deliberate trade-off. We sacrificed Precision (more false positives) to gain Recall (finding actual tumors), which is the correct engineering decision for a medical screening tool.

📂 Project Structure

DENSITY-AWARE-MAMMOGRAPHY/
│
├── app.py               # Flask Backend: Handles image uploads & inference
├── model.py             # PyTorch Architecture: Dual-Stream Network Code
├── dataset.py           # Custom Data Loader: Processes VinDr-Mammo images & CSVs
├── train.py             # Training Pipeline: Includes Weighted Loss logic
├── evaluate.py          # Evaluation Script: Generates Confusion Matrices
├── generate_plots.py    # Visualization: Creates research graphs for the UI
│
├── templates/
│   └── index.html       # The Frontend Interface (Rose-Themed)
│
├── static/
│   └── images/          # Stores generated plots (density_analysis.png, etc.)
│
├── data/                # (Excluded from Repo) Contains huge dataset images
└── requirements.txt     # Python dependencies


🛠️ Installation & Usage

1. Clone the Repository

git clone [https://github.com/muscaanmnmnm/Density-Aware-Mammography.git](https://github.com/muscaanmnmnm/Density-Aware-Mammography.git)
cd Density-Aware-Mammography


2. Install Dependencies

pip install -r requirements.txt


3. Setup the Dataset

Download the VinDr-Mammo dataset (PNG version) or use kagglehub (included in scripts).

Ensure finding_annotations.csv is in the data/ folder.

4. Run the Application

To launch the diagnostic web interface:

python app.py


Open your browser and go to: http://127.0.0.1:5000

5. Retrain the Model (Optional)

If you want to train the AI from scratch:

python train.py


🖼️ Interface Preview

The system features a professional, user-friendly interface designed for clinical interaction:

Secure Upload: Drag-and-drop mammogram analysis.

Clinical Context: Input fields for Density and View Position.

Research Dashboard: Visualizations of model performance and dataset statistics.
https://v0-breast-cancer-detection-app-jc.vercel.app/

🤝 Acknowledgments

Dataset: VinDr-Mammo

Frameworks: PyTorch, Flask, Tailwind CSS.