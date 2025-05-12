# Setup & Imports
import cv2
import numpy as np
import torch
import faiss
from PIL import Image
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from transformers import ViTModel, ViTFeatureExtractor
from torchcam.methods import GradCAM
from torchvision.transforms import ToTensor
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


# Object Detection with Mask R-CNN
def setup_mask_rcnn():
    """Load pre-trained Mask R-CNN model."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

def segment_objects(predictor, image_path):
    """Detect and segment objects in an image."""
    image = cv2.imread(Dog.jpg)
    outputs = predictor(image)
    masks = outputs["instances"].pred_masks.cpu().numpy()
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()
    return image, masks, boxes, classes

# Feature Extraction with ViT
def setup_vit():
    """Load pre-trained ViT model and feature extractor."""
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    return model, feature_extractor

def extract_features(model, feature_extractor, image, mask):
    """Extract ViT features for a masked object."""
    masked_image = image * mask[..., None]
    masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))
    inputs = feature_extractor(images=masked_image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Pooled features (768D)



    # FAISS Indexing for Similarity Search
def build_faiss_index(features):
    """Build a FAISS index for fast similarity search."""
    features = normalize(features, axis=1)  # L2 normalization
    index = faiss.IndexFlatIP(features.shape[1])  # Inner product (cosine similarity)
    index.add(features)
    return index

def query_faiss(index, query_features, k=5):
    """Query the FAISS index for top-k similar images."""
    query_features = normalize(query_features, axis=1)
    D, I = index.search(query_features, k)
    return D, I  # Distances and indices of top-k matches



    # Explainability with Grad-CAM

def apply_gradcam(model, image, target_layer="vit.encoder.layer.11"):
    """Generate Grad-CAM heatmap for ViT."""
    cam_extractor = GradCAM(model, target_layer)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    activation_map = cam_extractor(outputs.logits.argmax().item(), outputs)
    return activation_map[0].squeeze(0).numpy()

def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlay heatmap on the original image."""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(image, alpha, heatmap_colored, 1 - alpha, 0)
    return overlayed
    
