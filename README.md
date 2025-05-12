# 🔍 XAI-Based Content-Based Image Retrieval (CBIR) System using Mask R-CNN, ViT, and FAISS

This project is an advanced **Content-Based Image Retrieval (CBIR)** system that combines deep learning models and explainable AI (XAI) techniques to deliver accurate, interpretable image search results. It integrates:

- 🎭 **Mask R-CNN** for object instance segmentation
- 👁️ **Vision Transformer (ViT)** for powerful global image feature extraction
- ⚡ **FAISS** (Facebook AI Similarity Search) for fast and scalable nearest neighbor retrieval
- 🧠 **XAI** techniques (Grad-CAM / Attention Rollout) to visualize decision reasoning
- 📸 A full end-to-end CBIR pipeline: indexing, querying, feedback, and explanation

---

## 🚀 Features

- **Instance Segmentation**: Use Mask R-CNN to isolate key regions in input images
- **Transformer-Based Features**: Extract global and local descriptors using pretrained ViT models
- **Fast Retrieval**: Use FAISS for efficient similarity search over millions of embeddings
- **Explainability**: Generate attention maps and visual explanations for retrieved results
- **Flexible Input**: Query by image + textual prompt (e.g. “same but at night”)
- **Relevance Feedback Loop**: Improve retrieval by iteratively learning from user feedback

---

## 📂 Project Structure

