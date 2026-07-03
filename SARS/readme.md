# Guided Patient Segmentation and Semi-Supervised Label Propagation on SARS Data

This repository contains an advanced machine learning project focused on strategic health informatics and database segmentation. Instead of relying on traditional, unguided unsupervised learning, this project implements a **semi-supervised, metric-learning pipeline** to isolate, analyze, and propagate clinical outcome groups across large-scale epidemiological records.

## About the Dataset (Context for International Recruiters)
The project utilizes the public database of **Severe Acute Respiratory Syndrome (SARS)** provided by **SUS (Sistema Único de Saúde)**, Brazil's universal healthcare system. Serving over 200 million citizens, SUS microdata provides high-dimensional, sparse, and realistic medical records. Navigating this dataset requires tackling real-world data science challenges, such as heavily unbalanced classes and categorical structural noise.

## Project Overview & Methodology

The core objective of this project is to optimize patient segmentation by leveraging a smaller subset of validated clinical outcomes to guide the feature space transformation. 

Instead of traditional unsupervised clustering, which can struggle with high-dimensional categorical health data, this workflow utilizes **Supervised Distance Metric Learning** combined with an **unlabeled-to-labeled propagation pipeline**. This allows the model to learn the mathematical topology that best defines patient clinical outcomes and map those behaviors across the entire database.

## Core Features & Workflow

### 1. Advanced Preprocessing & Encoding
* Engineered a pipeline to handle high-cardinality categorical medical indicators and missing clinical fields.
* Formatted the feature matrices to isolate demographic parameters, underlying comorbidities, and incoming acute symptoms.

### 2. Metric Learning via Neighborhood Components Analysis (NCA)
* **Guided Dimensionality Reduction:** Implemented `NeighborhoodComponentsAnalysis` (NCA) to learn an optimal linear transformation of the feature space. 
* Unlike unsupervised techniques (like PCA), NCA maximizes the nearest-neighbor classification score in the transformed space using the guided validation labels, effectively stretching the axes that matter most for clinical outcomes and shrinking noise.

### 3. Semi-Supervised Label Propagation Pipeline
* Built an integrated `scikit-learn` Pipeline combining the learned NCA space transformation with a cost-sensitive `LogisticRegression` classifier (`class_weight="balanced"`).
* Used this specialized pipeline to learn decision boundaries from the guided subsets and accurately propagate classification labels across the dense, unlabeled segments of the main SARS database.

### 4. Evaluation & Clinical Insights
* Evaluated how well the propagated segments aligned with actual high-risk metrics (e.g., intensive care admission, respiratory failure levels, and mortality risk).
* Demonstrated that embedding guided metric learning drastically improves classification stability compared to unguided distance algorithms.

## Technologies & Frameworks

* **Data Manipulation & Preprocessing:** `pandas`, `numpy`
* **Machine Learning & Pipeline Architecture:** `scikit-learn` (`NeighborhoodComponentsAnalysis`, `LogisticRegression`, `Pipeline`)
* **Visualization:** `matplotlib`, `seaborn`

## Key Engineering Takeaways

* **Beyond Traditional Clustering:** By shifting from unguided clustering to an NCA-driven semi-supervised approach, the project bypassed the classical pitfalls of clustering high-dimensional categorical data.
* **Dimensional Efficiency:** NCA successfully compressed the clinical feature space into non-linear proxy dimensions that highly correlate with patient risk.
* **Scalable Labeling:** The label propagation architecture provides a scalable framework for clinical triage, allowing healthcare systems to quickly classify risk profiles of new incoming patients based on historical structural patterns.
