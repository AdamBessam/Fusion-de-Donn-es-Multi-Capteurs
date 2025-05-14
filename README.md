
# 🌱 PlantVision AI

**PlantVision AI** is an intelligent, web-based platform for the automated detection and classification of plant health, with a focus on identifying *Armillaria* fungal infections in cherry trees. By integrating multi-sensor data (RGB and multispectral images) with advanced AI techniques, the platform leverages state-of-the-art **Convolutional Neural Networks (CNNs)** to classify infection stages — enabling precise, early intervention and enhanced agricultural decision-making.

Developed as part of the final year project for the **Computer Engineering and Emerging Technologies (2ITE)** program at **ENSA El Jadida**, under the supervision of **Mr. Fahd Kalloubi**.

---

## 📑 Table of Contents

- [📖 Project Overview](#project-overview)
- [✨ Features](#features)
- [🛠️ Technologies Used](#technologies-used)
- [📁 Repository Structure](#repository-structure)
- [⚙️ Installation](#installation)
- [🚀 Running the Streamlit Application](#running-the-streamlit-application)
- [🖼️ Dataset](#dataset)
- [📝 Usage](#usage)
- [👥 Contributors](#contributors)
- [📄 License](#license)

---

## 📖 Project Overview

The goal of **PlantVision AI** is to create an interactive platform that leverages deep learning models for analyzing plant health through RGB and multispectral imagery. It incorporates cutting-edge CNN architectures — **ResNet**, **EfficientNet**, **Xception**, and **DenseNet** — and implements both **early** and **late fusion strategies** to maximize classification accuracy.

Built with **Streamlit**, the platform allows users to:
- Upload images
- Select analysis modalities
- View infection stage predictions with corresponding confidence scores

---

## ✨ Features

- 📤 **Image Upload**: Supports uploading of RGB images, multispectral images, or both.
- 🔍 **Model Selection**: Choose between unimodal (RGB or multispectral) and multimodal (fusion of both) models.
- 🌿 **Infection Stage Classification**: Classifies plant health into:
  - Healthy 🌳
  - Stage 1 (Early Infection) 🌱
  - Stage 2 (Moderate Infection) 🍂
  - Stage 3 (Advanced Infection) 🍁
- 🎛️ **Interactive Interface**: Streamlit-powered, user-friendly web UI.
- 📊 **Performance Metrics**: Displays prediction results with confidence levels.
- 📈 **Scalable Architecture**: Modular design allows easy integration of new diseases or sensors.

---

## 🛠️ Technologies Used

| Category             | Tools & Frameworks                     |
|:---------------------|:---------------------------------------|
| **Programming Language** | Python                              |
| **Web Framework**        | Streamlit                           |
| **Machine Learning**     | PyTorch, NumPy, OpenCV               |
| **Development Tools**    | Visual Studio Code, Google Colab, Google Drive |
| **Version Control**      | Git, GitHub                          |
| **Project Management**   | Trello (Scrum methodology)           |
| **Styling**              | CSS                                   |

---

## 📁 Repository Structure

```
plateforme/
│
├── app.py                    # Point d'entrée de l'application
├── style.css                 # Feuille de style principale
├── .gitattributes            # Configuration Git
│
├── assets/
│   └── style.css             # CSS complémentaire ou spécifique à certains éléments
│
├── components/               # Contient les modules de traitement et modèles
│   ├── cnn_model.py
│   ├── densenet_model.py
│   ├── efficientnet_model.py
│   ├── resnet_model.py
│   ├── xception_model.py
│   ├── prediction.py         # Fonctions de prédiction
│   ├── preprocessing.py      # Prétraitement des données
│   ├── ui.py                 # Interface utilisateur (peut-être pour Streamlit)
│   └── __pycache__/          # Fichiers Python compilés (.pyc)
│
├── config/                   # Fichiers de configuration
│   ├── settings.py
│   └── __pycache__/
│
└── models/                   # Contient les modèles entraînés
    ├── model_loader.py       # Chargement des modèles
    ├── multimodal/           # Modèles traitant plusieurs sources d'entrée
    │   ├── best_model.pth
    │   ├── best_model_densenet.pth
    │   ├── ...
    ├── unimodal/             # Modèles traitant une seule source d'entrée
    │   ├── best_custom_cnn_model.h5
    │   ├── best_model_resnet.pth
    │   ├── ...
    └── __pycache__/

```

---

## ⚙️ Installation

To set up the project locally:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/username/plantvision-ai.git
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Pre-trained Models**
   Ensure pre-trained model files are available under:
   ```
   models/multimodal/
   models/unimodal/
   ```
   📩 *Contact the project supervisor if model files are missing.*

---

## 🚀 Running the Streamlit Application

To launch the app locally:

1. **Check Dependencies**
   Ensure all required packages from `requirements.txt` are installed.

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**
   Navigate to: [http://localhost:8501](http://localhost:8501)

4. **Usage Instructions**
   - Select analysis type (Unimodal or Multimodal) via sidebar.
   - Choose appropriate model (e.g., ResNet18, EfficientNetB0).
   - Upload images.
   - View prediction results with confidence scores.

---

## 🖼️ Dataset

The dataset comprises RGB and multispectral images of cherry trees, collected between **July 2021 - July 2022** in a Macedonian orchard.

- **RGB Images**: Standard visible spectrum images.
- **Multispectral Images**: Includes Green, Red, NIR, and RedEdge bands.
- **Annotations**: Expert-labeled into:
  - Healthy
  - Stage 1
  - Stage 2
  - Stage 3
- **Samples**: 577 trees (514 Healthy, 51 Stage 1, 7 Stage 2, 5 Stage 3)

📝 *Note: Dataset is not publicly available. Contact project supervisor for access.*

---

## 📝 Usage

1. **Select Analysis Mode**
   - *Unimodal*: RGB or multispectral analysis.
   - *Multimodal*: Combined analysis with fusion strategy selection.

2. **Upload Images**
   - RGB: Upload a single RGB image.
   - Multispectral: Upload a folder of multispectral bands.
   - Multimodal: Upload both.

3. **View Results**
   - Predicted health status and confidence score displayed.

---

## 👥 Contributors

| Name               | Role              |
|:------------------|:-----------------|
| **Siham DRIHAM**   | Developer         |
| **Adam BESSAM**    | Developer         |
| **Mohammed DAOUDI**| Developer         |
| **Walid CHOUAY**   | Developer         |
| **Fahd KALLOUBI**  | Project Supervisor|

---

✅ *Feel free to fork, star ⭐, and contribute!*
