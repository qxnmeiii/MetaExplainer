# MetaExplainer
Based on the structure and style of the provided example, here is the documentation for **MetaExplainer** using the content from the uploaded PDF.

***

# MetaExplainer: Revisit Domain Generalization of Functional Connectome Analyses from The Perspective of Explainability

## 🔍 Description

**MetaExplainer** is a dedicated **domain generalization (DG)** framework for **fMRI-based neuropsychiatric disorder diagnosis**. It addresses a critical gap in existing Graph Neural Network (GNN) studies—which typically treat explainability and generalizability independently—by unifying them under a meta-learning paradigm.

The core premise of MetaExplainer is that **cross-domain generalizability** relies on the ability to stably capture subtle, disease-related **explanation factors** (connectome biomarkers) across heterogeneous clinical centers. By prioritizing the learning of these domain-agnostic, neurobiologically plausible patterns, the framework achieves:

* **Task-Oriented Network Construction:** A **Multi-Head Self-Attention (MHSA)** module constructs personalized, nonlinear functional connectivity (FC) networks directly from BOLD signals.
* **Explainable Generalization:** Novel **Explanatory-Generalizability Regularizations (EGRs)** enforce the sparsity and cross-site stability of group-wise connectome differences.
* **Robust Diagnosis:** A **Dual-Loop Meta-Learning** algorithm simulates domain shift scenarios during training to ensure the learned biomarkers are robust to inter-site heterogeneity.

Evaluated on large-scale multi-center datasets (**ABIDE** for ASD and **REST-meta-MDD** for MDD), MetaExplainer achieves state-of-the-art performance (AUC: 76.32% for ASD, 65.31% for MDD) and identifies biomarkers consistent with clinical literature.

---

## 🧭 Framework Overview

The **MetaExplainer** pipeline consists of two fundamental components: an end-to-end fMRI-based GNN model for diagnosis and an explanatory-generalizability meta-learning algorithm for training.


---

## 🗂 Step 1. Data Preparation

### 🔹 Multi-Center Datasets

The framework is evaluated on two major publicly available benchmarks.

* **ABIDE (Autism Brain Imaging Data Exchange):** Comprises resting-state fMRI (rs-fMRI) from 17 international sites, including 416 ASD patients and 418 typically developing controls.
* **REST-meta-MDD:** A large-scale consortium dataset from 17 hospitals in China, containing 1300 MDD patients and 1128 healthy controls across 25 cohorts.

### 🔹 Preprocessing Pipeline

To ensure consistency, standard preprocessing pipelines were utilized:

1. **Preprocessing:** Data underwent slice-timing correction, motion realignment, and band-pass filtering (using CPAC for ABIDE and DPARSF for REST-meta-MDD).
2. **Parcellation:** The brain was parcellated into **200 Regions of Interest (ROIs)** using the **CC200 atlas**.
3.  **Feature Extraction:** The average BOLD signal was extracted for each ROI. For the diagnostic module initialization, Pearson correlation coefficients were calculated as initial node features.

---

## 🧠 Step 2. fMRI-Based GNN Model

MetaExplainer learns to construct and analyze brain networks in an end-to-end fashion, moving beyond pre-defined linear correlations.

### 🔹 MHSA-Based Brain Network Constructor

Given raw BOLD signals $X \in \mathbb{R}^{n \times T}$, this module constructs a task-oriented, nonlinear graph $G=(V, E, M)$:

1.  **Feature Embedding:** 1D Convolutional layers ($Conv1d \rightarrow BN \rightarrow ReLU$) map temporal BOLD signals into a nonlinear representation space.
2.  **Graph Construction:** A **Multi-Head Self-Attention (MHSA)** mechanism aggregates cross-ROI associations to update ROI embeddings. The learned Functional Connectivity (FC) matrix $M$ is defined as $M=AA^T$, where $A$ is the concatenated output of the MHSA heads.

### 🔹 GCN-Based DiagNet

The constructed graph is fed into a **Graph Convolutional Network (GCN)** classifier:

* **Architecture:** Three GCN layers with progressive dimensionality reduction ($200 \rightarrow 64 \rightarrow 8$ units) to capture hierarchical connectivity patterns.
* **Update Rule:** Node features are updated via $g_{i} = BN_{i}(LeakyReLU(Mg_{i-1}U_{i}))$.

---

## 🚀 Step 3. Explanatory-Generalizability Meta-Learning

To ensure the learned FC networks are domain-agnostic, the model is trained using a dual-loop meta-learning algorithm guided by specific regularizations.

### 🔹 Explanatory-Generalizability Regularizations (EGRs)

Two clinical knowledge-inspired constraints are applied to the inter-group FC differences ($\Delta_{D_{k}}$) to capture stable biomarkers:

1.  **Sparsity ($L_{sp}$):** Based on the assumption that early-stage disorder changes are regionalized, an entropy-based penalty enforces sparsity in group-wise differences:
    $L_{sp}=-||\sum_{k}\Delta_{D_{k}}\odot log_{2}\Delta_{D_{k}}||_{2}$
2.  **Stability ($L_{cons}$):** Ensures that disease-related FC differences remain consistent across different clinical centers ($D_i, D_j$):
    $L_{cons} = 1 - \frac{2 \lVert \Delta_{D_{i}} \odot \Delta_{D_{j}} \rVert_{2}}{\lVert \Delta_{D_{i}} \rVert_{2} + \lVert \Delta_{D_{j}} \rVert_{2}}$

### 🔹 Dual-Loop Algorithm

The training process simulates domain generalization scenarios:

* **Inner-Loop (Adaptation):** The model is trained on a subset of source domains ($S_{inner}$) by minimizing the standard classification loss ($L_{ce}$) to adapt to domain-specific variations.
* **Outer-Loop (Generalization):** The model is updated based on its performance on the remaining source domains ($S_{outer}$), optimizing a meta-objective that integrates the EGRs:
    $$L_{meta}=L_{ce}+\alpha L_{sp}+\beta L_{cons}$$

---

## 💾 Code and Data Availability

The datasets used in this study are publicly available, and the source code is committed for release upon publication.

* **ABIDE Dataset:** [http://preprocessed-connectomes-project.org/abide/](http://preprocessed-connectomes-project.org/abide/)
* **REST-meta-MDD Dataset:** [https://rfmri.org/REST-meta-MDD](https://rfmri.org/REST-meta-MDD)
* **Codebase:** The authors state, "We commit to releasing the complete source code and implementation details upon publication to ensure full reproducibility".
