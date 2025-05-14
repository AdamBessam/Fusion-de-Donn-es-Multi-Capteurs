import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import torchvision.models as models
import traceback
from config.settings import CLASSES

import traceback
import os

class ModelWrapper:
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        
    def __getattr__(self, name):
        return getattr(self.model, name)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

@st.cache_resource
def load_multispectral_model(approach="cnn"):
    try:
        approach = approach.lower()
        
        model_path = None
        
        if approach == "cnn":
            model_path = "models/unimodal/best_custom_cnn_model.h5"
            
        elif approach == "lightweight":
            model_path = "models/unimodal/lightweight-cnn.pth"
            
        elif approach == "mobilenetv3small":
            model_path = "models/unimodal/best_model_MultispectralMobileNetV3Small.pth"
            
        elif approach == "efficientnetb0":
            model_path = "models/unimodal/best_model_MultispectralEfficientNetB0.pth"
            
        elif approach == "densenet121":  
            model_path = "models/unimodal/best_model_MultispectralDenseNet121.pth"
            
        elif approach == "resnet18":
            model_path = "models/unimodal/best_model_MultispectralResNet18.pth"
            
        else:
            st.error(f"Approche '{approach}' non reconnue.")
            return None
        
        if not os.path.exists(model_path):
            st.error(f"Fichier modèle non trouvé: {model_path}")
            return None
        
        if approach == "cnn":
            model = load_model(model_path)
            model_type = "keras"
            
        else:  
            loaded_data = torch.load(model_path, map_location=torch.device('cpu'))
            
            if isinstance(loaded_data, dict):
                st.warning(f"Le fichier {model_path} contient un état de modèle (state_dict). "
                          "Le modèle sera utilisé comme un dictionnaire.")
            
            model = loaded_data
            
            if hasattr(loaded_data, 'eval'):
                model.eval()
                
            model_type = "pytorch"
            
        st.info(f"Modèle {approach} chargé depuis {model_path}")
        
        if model_type == "keras":
            st.info(f"Forme d'entrée attendue: {model.input_shape}")
            st.info(f"Forme de sortie du modèle: {model.output_shape}")
        elif model_type == "pytorch":
            st.info(f"Modèle PyTorch chargé: {type(model).__name__}")
        
        if model_type == "keras":
            model.model_type = model_type
            return model
        else:
            return ModelWrapper(model, model_type)
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle {approach}: {e}")
        st.error(f"Détails: {traceback.format_exc()}")
        return None

@st.cache_resource
def load_rgb_model(approach="resnet50"):
    try:
        model_paths = {
            "resnet50": "models/unimodal/cherry_tree_resnet_pytorch.pth",
            "resnet18": "models/unimodal/best_model_resnet.pth",
            "efficientnetb0": "models/unimodal/best_model_EfficientNetB0.pth",
        }
        
        if approach.lower() not in model_paths:
            st.warning(f"Approche '{approach}' non reconnue pour le modèle RGB. Utilisation de 'resnet50' par défaut.")
            approach = "resnet50"
        
        model_path = model_paths[approach.lower()]
        if not os.path.exists(model_path):
            st.error(f"Le fichier modèle RGB pour l'approche '{approach}' n'existe pas: {model_path}")
            return None
        
        if approach.lower() == "resnet50":
            model = models.resnet50(pretrained=False)
            
            for param in model.parameters():
                param.requires_grad = False
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(CLASSES))
            )
        
        elif approach.lower() == "resnet18":
            model = models.resnet18(pretrained=False)
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(CLASSES))
        
        elif approach.lower() == "efficientnetb0":
            try:
                from torchvision.models import efficientnet_b0
                model = efficientnet_b0(pretrained=False)
                
                num_ftrs = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(num_ftrs, len(CLASSES))
                )
            except (ImportError, AttributeError):
                st.warning("EfficientNetB0 n'est pas disponible dans cette version de torchvision. Tentative de chargement direct du modèle.")
                model = torch.load(model_path, map_location=torch.device('cpu'))
                if isinstance(model, dict):  
                    st.error("Le fichier chargé est un état de modèle (state_dict) et non un modèle complet. Architecture EfficientNetB0 non disponible.")
                    return None
        
        if approach.lower() in ["resnet50", "resnet18", "efficientnetb0"]:
            try:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                if isinstance(state_dict, dict) and not isinstance(state_dict, nn.Module):
                    model.load_state_dict(state_dict, strict=False)  
                else:
                    st.warning(f"Le fichier pour {approach} contient un modèle complet au lieu d'un state_dict. Extraction de l'état...")
                    if hasattr(state_dict, 'state_dict'):
                        model.load_state_dict(state_dict.state_dict(), strict=False)
                    else:
                        model = state_dict
            except Exception as e:
                st.error(f"Erreur lors du chargement des poids pour {approach}: {e}")
                st.error(f"Tentative de chargement direct du modèle...")
                model = torch.load(model_path, map_location=torch.device('cpu'))
        
        model.eval()
        
        st.info(f"Modèle RGB {approach} chargé avec succès")
        return ModelWrapper(model, "pytorch")  
    
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle RGB {approach}: {e}")
        st.error(f"Détails: {traceback.format_exc()}")
        return None

@st.cache_resource
def load_multimodal_early_fusion_model(approach="normal"):
    """
    Charge le modèle de fusion précoce selon l'approche spécifiée
    """
    try:
        model_paths = {
            "cnn": "models/multimodal/best_model_multimodal_cnn.pth",
            "efficientnet": "models/multimodal/best_model_efficientnet.pth",
            "resnet": "models/multimodal/best_model_resnet.pth",
            "xception": "models/multimodal/best_model_xception.pth",
            "densenet": "models/multimodal/best_model_densenet.pth"
        }
        
        model_path = model_paths[approach]
        if not os.path.exists(model_path):
            st.warning(f"Le fichier modèle pour l'approche '{approach}' n'existe pas. Utilisation du mode simulation.")
            return None

        if approach == "densenet":
            from components.densenet_model import load_densenet_multimodal_model
            return load_densenet_multimodal_model(model_path)
        elif approach == "efficientnet":
            from components.efficientnet_model import load_efficientnet_multimodal_model
            return load_efficientnet_multimodal_model(model_path)
        elif approach == "resnet":
            from components.resnet_model import load_resnet_multimodal_model
            return load_resnet_multimodal_model(model_path)
        elif approach == "xception":
            from components.xception_model import load_xception_multimodal_model
            return load_xception_multimodal_model(model_path)
        elif approach == "cnn":
            from components.cnn_model import load_cnn_multimodal_model
            return load_cnn_multimodal_model(model_path)
    
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle de fusion précoce ({approach}): {e}")
        st.error(f"Détails: {traceback.format_exc()}")
        return None