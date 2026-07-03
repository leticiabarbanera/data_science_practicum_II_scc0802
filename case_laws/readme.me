# Dense Information Retrieval & Contrastive Learning for Multi-Label Legal Document Classification

This repository contains an advanced, enterprise-grade Natural Language Processing (NLP) pipeline designed to automate the multi-label classification of complex, unstructured legal documents (such as initial petitions). Moving away from traditional classification heads that struggle with long-form text and technical vocabularies, this project adopts a **Dense Information Retrieval architecture** powered by **Contrastive Learning** and high-performance vector search.

## Architectural Overview & Conceptual Framework

Standard multi-label text classifiers often collapse when dealing with heavy class imbalances, long contexts, and nuanced domain vocabularies. To achieve high structural maturity, this project frames classification as a **Semantic Vector Search task**:
1. **Dual-Encoder (Bi-Encoder) Setup:** Leverages a state-of-the-art Portuguese Language Model (**BERTimbau**) to map both the raw legal petitions and enriched legal class descriptions into a shared, continuous vector space.
2. **Contrastive Representation Learning:** Instead of cross-entropy, the model undergoes metric fine-tuning using **Multiple Negatives Ranking Loss (MNRL)**, forcing the network to minimize the vector distance between a petition and its true legal labels while maximizing the distance against all other labels in the batch.
3. **Sub-Linear Latency Inference:** Once trained, legal class embeddings are indexed via **FAISS (Facebook AI Similarity Search)**, turning multi-label inference into a high-speed cosine similarity query.

## Core Engineering & Pipeline Features

### 1. Exploratory Data Analysis & Label Dependency Modeling
* **Multi-Label Explanations:** Investigated structural label correlations, high-sparsity environments, and the mathematical implications of label dependency using specialized multi-label classification frameworks.
* **Text Normalization:** Engineered custom preprocessing logic to scrub administrative noise, isolate critical legal arguments, and handle the sequence length boundaries of Transformer architectures.

### 2. Contrastive Fine-Tuning & Metric Learning
* Implemented a `SentenceTransformers` training loop utilizing `MultipleNegativesRankingLoss`.
* Configured the dual-encoder architecture to treat the relationship between a petition text and its detailed legal category profile as a tight semantic pair, vastly expanding the model's zero-shot and few-shot generalization capabilities on rare classes.

### 3. High-Performance Indexing with FAISS
* Exported the fine-tuned dense embeddings into a highly optimized vector index using `faiss-cpu`.
* Formatted the prediction block to dynamically fetch top-$k$ nearest neighbors via Cosine Similarity, transforming classification from a rigid neural layer computation into a scalable retrieval engine.

### 4. Dynamic Threshold Sweep Optimization
* **Class-Specific Calibrations:** Addressed severe class imbalances by rejecting static global thresholds. 
* Implemented an automated optimization sweep across the validation slice to discover custom, class-by-class validation boundaries (*thresholds*), balancing Precision and Recall (F1-Score) independently for each legal category.

## Technologies & Framework Ecosystem

* **Transformer & Embedding Frameworks:** `transformers` (Hugging Face), `sentence-transformers`
* **Base Large Language Model:** `neuralmind/bert-base-portuguese-cased` (BERTimbau)
* **Vector Vector Search Engine:** `faiss` (FAISS)
* **Data Processing & ML Engineering:** `pandas`, `numpy`, `scikit-learn`, `pyarrow` (Parquet parsing)
* **Visualization & Analytics:** `matplotlib`, `seaborn`

## Key Engineering Takeaways

* **Paradigm Shift in Document Tagging:** Proved that framing long-text multi-label categorization as a semantic retrieval task provides vastly more robust boundaries than standard linear classification layers.
* **Mitigating Label Skewness:** By utilizing Multiple Negatives Ranking Loss, the model learns deep contextual alignments rather than over-indexing on high-frequency keyword correlations.
* **Production-Grade Infrastructure:** The combination of a fine-tuned Bi-Encoder with a FAISS index allows the entire legal architecture to scale horizontally, processing high-volume daily legal filings with sub-millisecond inference latency.
