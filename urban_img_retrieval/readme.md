# Multimodal Ensemble & Topographic Re-ranking for Advanced Image Retrieval

This repository contains an advanced computer vision and information retrieval pipeline developed for large-scale image-to-image and text-to-image search tasks. The core architecture implements a hybrid multi-model embedding ensemble combined with transductive re-ranking and pseudo-relevance feedback algorithms to maximize mean average precision metrics.

## Project Overview & Strategic Architecture

Modern information retrieval pipelines often fail when relying on a single visual descriptor. This project circumvents that limitation by leveraging an **ensemble of disparate Foundation Models**:
1. **SigLIP (Sign Language-Image Pre-training):** Captures robust semantic, contrastive text-image alignment and compositional scene understandings.
2. **DINOv2:** Captures granular, self-supervised dense visual features, geometric properties, and object-part correspondences independent of textual definitions.

By fusing these latent representations into an optimized joint similarity space and applying **pseudo-classification re-ranking (local neighborhood refinement)**, the pipeline effectively corrects retrieval drift and bridges the semantic-structural gap.

## Core Engineering & Algorithmic Workflow

### 1. Multi-Model Feature Extraction (Foundation Models)
* **Zero-Shot Representations:** Built robust inference pipelines using PyTorch to extract continuous deep feature vectors from large-scale image galleries and queries.
* **Dual-Topology Embedding:** Harmonized contrastive language-vision embeddings from Google's `SigLIP` with self-supervised visual descriptors from Meta's `DINOv2`.

### 2. Matrix Algebra Fusion & Similarity Optimization
* Generated dense multi-gigabyte cosine similarity matrices tracking the geometric alignment between query vectors and gallery pools.
* **Hyperparameter-Driven Fusion:** Engineered an parameterized blending mechanism ($Similarity_{Final} = \alpha \cdot Sim_{SigLIP} + (1-\alpha) \cdot Sim_{DINOv2}$) to dynamically calibrate the influence of textual-semantic vs. purely visual-structural patterns.

### 3. Pseudo-Relevance Feedback & Neighborhood Re-ranking
* **Pseudo-Labeling Strategies:** Implemented a `pseudo_top_k` algorithm that treats high-confidence top-ranked gallery assets as surrogate targets to refine the original query vectors.
* **Local Graph Re-ranking:** Enforced local manifold constraints via an optimized reranking pool window ($k$-reciprocal style neighbors) to down-weight false positives and surface structurally isolated true matches.

### 4. Bayesian/Local Grid Search Optimization
* Integrated automated optimization strategies using validation boundaries to iteratively converge on the best structural weights (`BEST_ALPHA`), local neighborhood scopes (`BEST_RERANK_POOL_K`), and activation thresholds (`BEST_MIN_CONFIDENCE`).
* Designed the final inference block to export high-fidelity prediction maps formatted for enterprise submission benchmarks.

## Technologies & Deep Learning Ecosystem

* **Core Deep Learning Framework:** `torch` (PyTorch), `torchvision`
* **Pre-trained Architecture Hubs:** `transformers` (Hugging Face), `timm` (Torch Image Models)
* **High-Performance Matrix Processing:** `numpy`, `pandas`, `scipy`
* **Data Pipelines & Imaging:** `PIL` (Pillow), `pathlib`, `tqdm`

## Key Engineering Takeaways & Framework Strengths

* **Cross-Model Complementarity:** The architecture proves that contrastive vision-language models and self-supervised feature extractors yield non-overlapping error profiles. Fusing their geometric spaces drastically expands structural recall.
* **Manifold Alignment via Re-ranking:** Applying pseudo-relevance feedback corrected for vector space distortions, pushing true structural matches higher up the retrieval ranking curve even under initial low-confidence conditions.
* **Production-Ready Inference:** The pipeline operates strictly via vectorized tensor operations, allowing the entire multi-thousand asset gallery mapping and re-ranking matrix to be executed with minimized computational footprint.
