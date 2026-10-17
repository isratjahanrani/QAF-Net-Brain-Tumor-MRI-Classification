# 🧠 QAF-Net: Quantization-Aware Federated Learning with Site-Local Normalization for Cross-Dataset Brain Tumor MRI Classification on Edge Devices

[![Paper](https://img.shields.io/badge/Research%20Paper-PDF-red.svg)](Group_01_Brain_Tumor_Classification_and_Detection_using_TinyML.pdf)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-TinyML%20%7C%20606%20KB%20INT8-brightgreen.svg)](#tinyml-deployment-envelope--edge-benchmarks)
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-FedAvg%20%7C%20FedProx%20%7C%20FedBN-blue.svg)](#federated-architecture--qaf-net-methodology)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras%20%7C%20TFLite-orange.svg)](https://www.tensorflow.org)
[![Institution](https://img.shields.io/badge/Institution-United%20International%20University%20(UIU)-purple.svg)](https://www.uiu.ac.bd)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-isratjahanrani-blue.svg)](https://github.com/isratjahanrani)

An end-to-end medical image processing and privacy-preserving TinyML research framework for cross-institutional **brain tumor MRI classification on resource-constrained clinical edge hardware**. 

Developed in the **Department of Computer Science and Engineering** at **United International University (UIU)**, this project demonstrates **QAF-Net** — a federated learning architecture that overcomes scanner-specific domain shifts between global cohorts (**CHENG - China** and **PMRAM - Bangladesh**). By decontaminating aggregated benchmarks (**uncovering 55.2% near-duplicate leakage**) and dynamically re-estimating site-local batch normalization statistics from unlabeled target scans, QAF-Net achieves **94.7% held-out test accuracy** within a **606 KB INT8 quantized model footprint**.

---

## 📌 Table of Contents

- [Clinical Motivation & Research Challenges](#clinical-motivation--research-challenges)
- [Summary of Core Contributions](#summary-of-core-contributions)
- [The Benchmark Deduplication Audit (55.2% Contamination Discovery)](#the-benchmark-deduplication-audit-552-contamination-discovery)
- [Multi-Center Hospital Cohorts & Acquisition Profiles](#multi-center-hospital-cohorts--acquisition-profiles)
- [QAF-Net Methodology & Clinical Federated Architecture](#qaf-net-methodology--clinical-federated-architecture)
  - [1. Site-Local Normalization for Scanner Adaptation](#1-site-local-normalization-for-scanner-adaptation)
  - [2. Per-Channel INT8 Quantization Regularization](#2-per-channel-int8-quantization-regularization)
  - [3. Superset Loss for Incomplete Clinical Labels](#3-superset-loss-for-incomplete-clinical-labels)
- [Experimental Results (Leave-One-Site-Out Validation)](#experimental-results-leave-one-site-out-validation)
- [TinyML Edge Deployment & Resource Footprint](#tinyml-edge-deployment--resource-footprint)
- [Interpretability & Diagnostic Saliency (Grad-CAM)](#interpretability--diagnostic-saliency-grad-cam)
- [Four-Stage Notebook Pipeline Architecture](#four-stage-notebook-pipeline-architecture)
- [Repository Structure & Model Checkpoints](#repository-structure--model-checkpoints)
- [Setup & Reproduction Guide](#setup--reproduction-guide)
- [Authors & Academic Context](#authors--academic-context)
- [License & Acknowledgments](#license--acknowledgments)

---

## 🏥 Clinical Motivation & Research Challenges

Automated diagnostic screening of neuroimaging (MRI) holds immense potential to assist clinicians and radiologists in low-resource medical facilities. However, clinical implementation is hindered by three compounding obstacles:
1. **Patient Data Privacy & Regulatory Restrictions:** Patient MRI scans are strictly protected and cannot be transmitted to external servers or cloud repositories.
2. **Resource Scarcity in District Clinics:** Peripheral healthcare centers lack high-end GPU workstations and require models that can run on low-cost diagnostic tablets or embedded edge microcontrollers.
3. **Cross-Center Scanner Domain Shifts:** Magnetic resonance imaging varies dramatically across manufacturers (Siemens, GE, Philips), field strengths (1.5T vs. 3.0T), slice protocols (contrast-enhanced T1, T2, FLAIR), and operator configurations.

When federated learning is trained across distinct hospitals, the resulting global weights often exhibit severe sensitivity to post-training quantization. **QAF-Net** was designed to harmonize cross-scanner variations while ensuring near-lossless INT8 compression for edge deployment.

---

## 🏆 Summary of Core Contributions

- **🔍 C1: Discovery of 55.2% Duplicate Leakage in Popular Benchmarks:** Performed a systematic pairwise perceptual similarity audit demonstrating that more than half of the images in widely used Kaggle brain tumor benchmarks are near-duplicates of primary sources. We construct and release the first clean, deduplicated multi-site federation.
- **📈 C2: Scanner Domain Bridging via Site-Local Normalization:** Demonstrated that batch normalization statistics directly encode scanner-specific signal intensity transfers. Recalibrating BN parameters using a small batch of unlabeled local scans boosts held-out accuracy from **76.7% to 93.3%** and dramatically stabilizes model predictions across seeds.
- **⚡ C3: Production-Ready 606 KB TinyML Edge Model:** Deployed a fully integer-quantized INT8 MobileNet model of **606 KB**, fitting within **< 1 MB SRAM/Flash** requirements and incurring only a **0.15-point drop** from full FP32 precision.
- **🌐 C4: Flexible Superset Loss Formulation:** Engineered an objective function capable of federating disparate medical centers regardless of whether local datasets label tumors into granular subtypes (*glioma, meningioma, pituitary*) or simply binary flags (*tumor vs. normal*).

---

## 🔍 The Benchmark Deduplication Audit (55.2% Contamination Discovery)

A widespread issue in computer-aided medical diagnosis is testing algorithms on benchmark datasets that contain duplicated slices across training and validation splits:

| Dataset / Cohort | Total Slices | Inter-Cohort Duplicates | Intra-Cohort Near-Copies | Decontaminated Slices | Contamination Level |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SARTAJ** | 3,264 | 2,291 (derived from CHENG) | 425 (internal copies) | **544** | **83.3% Contaminated** |
| **Aggregated Kaggle** | 7,023 | 3,877 (cross-source copies) | 812 (near-copies) | **2,334** | **55.2% Contaminated** |
| **CHENG (China)** | 3,064 | 0 | 0 | **3,064** | **Clean (Zero Leakage)** |
| **PMRAM (Bangladesh)** | 1,514 | 0 | 0 | **1,514** | **Clean (Zero Leakage)** |

> [!IMPORTANT]
> By eliminating cross-source contamination, our leave-one-site-out benchmarks reflect genuine clinical generalization to unseen hospitals rather than artificial memorization of duplicated patient slices.

---

## 🏥 Multi-Center Hospital Cohorts & Acquisition Profiles

The research utilizes four distinct clinical imaging cohorts:
1. **CHENG Cohort (China):** 3,064 contrast-enhanced T1-weighted axial/coronal/sagittal slices across 233 patients from Nanfang Hospital and General Hospital of Tianyou. Grouped strictly patient-wise.
2. **PMRAM Cohort (Bangladesh):** 1,514 slices from brain cancer patients collected across medical centers in Bangladesh, exhibiting unique scanner noise profiles, varied slice intervals, and mixed T2/FLAIR sequences.
3. **SARTAJ Clean Cohort:** 544 verified unique slices containing healthy controls and balanced subtype representations.
4. **BR35H Cohort:** Binary clinical benchmark providing verified normal non-tumor scans.

---

## ⚙️ QAF-Net Methodology & Clinical Federated Architecture

```
                    ┌─────────────────────────────────┐
                    │      Central Federation Hub     │
                    │      Model Parameter Aggregator │
                    └────────────────┬────────────────┘
                                     │ Weights ϕ_t
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
     ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
     │ Site A: CHENG    │   │ Site B: PMRAM    │   │ Site C: SARTAJ   │
     │ 3,064 Slices     │   │ 1,514 Slices     │   │ 544 Slices       │
     ├──────────────────┤   ├──────────────────┤   ├──────────────────┤
     │ Site-Local BN    │   │ Site-Local BN    │   │ Site-Local BN    │
     │ INT8 Regularizer │   │ INT8 Regularizer │   │ INT8 Regularizer │
     └──────────────────┘   └──────────────────┘   └──────────────────┘
```

### 1. Site-Local Normalization for Scanner Adaptation
Rather than computing a global average of batch normalization parameters (which washes out scanner-specific intensity characteristics), each hospital retains its own BN statistics. For deployment to a new hospital, a small calibration batch of **10–20 unlabeled patient scans** is used to re-estimate the running mean and variance prior to inference.

### 2. Per-Channel INT8 Quantization Regularization
To prepare neural weights for embedded integer quantization on microcontrollers, local client training incorporates an INT8 grid proximity penalty:
$$\mathcal{L}_{\text{local}}(\mathbf{w}) = \mathcal{L}_{\text{task}}(\mathbf{w}) + \lambda \sum_{c} \left\| \mathbf{w}_c - \text{round}\left(\frac{\mathbf{w}_c}{\Delta_c}\right)\Delta_c \right\|_2^2$$

### 3. Superset Loss for Incomplete Clinical Labels
Accommodates hospital partners where only high-level diagnostic codes are logged:
$$\mathcal{L}_{\text{superset}} = - \log \sum_{k \in \mathcal{S}_{\text{tumor}}} P(y = k \mid \mathbf{x})$$

---

## 📊 Experimental Results (Leave-One-Site-Out Validation)

Multi-seed evaluation across leave-one-site-out (LOSO) clinical folds (mean $\pm$ standard deviation over 3 seeds):

| Federated Methodology | Held-Out Cohort | FP32 Accuracy (%) | INT8 Accuracy (%) | Quantization Loss $\Delta_q$ | Subtype Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized Upper Bound** | CHENG | $92.4 \pm 0.8\%$ | $92.1 \pm 0.7\%$ | $0.3 \pm 0.1$ | $92.1\%$ |
| **Local-Only Baseline** | CHENG | $41.0\%$ | $39.8\%$ | $1.2$ | $39.8\%$ |
| **FedAvg** | CHENG | $76.7 \pm 17.7\%$ | $75.5 \pm 16.9\%$ | $1.2 \pm 0.8$ | $75.5\%$ |
| **FedBN** | CHENG | $92.8 \pm 2.4\%$ | $92.2 \pm 2.1\%$ | $0.6 \pm 0.3$ | $92.2\%$ |
| **FedProx** | CHENG | $93.1 \pm 2.1\%$ | $92.8 \pm 1.9\%$ | $0.3 \pm 0.2$ | $92.8\%$ |
| **QAF-Net (Ours)** | **CHENG** | **$94.8 \pm 1.2\%$** | **$94.7 \pm 1.1\%$** | **$0.15 \pm 0.14$** | **$94.7\%$** |
| | | | | | |
| **FedAvg (Recalibrated)** | PMRAM | $77.8 \pm 2.1\%$ | $77.3 \pm 1.9\%$ | $-0.55 \pm 1.13$ | $77.3\%$ |
| **FedBN (Recalibrated)** | PMRAM | $79.1 \pm 2.5\%$ | $78.3 \pm 2.3\%$ | $0.8 \pm 0.4$ | $78.3\%$ |
| **QAF-Net (Recalibrated)** | **PMRAM** | **$78.9 \pm 2.0\%$** | **$77.8 \pm 2.1\%$** | $1.1 \pm 0.6$ | **$77.8\%$** |

---

## 📦 TinyML Edge Deployment & Resource Footprint

Optimized for on-device medical decision support without cloud latency or exposure:

```
FP32 Full-Precision Size:    ~2.40 MB
─────────────────────────────────────────────
QAF-Net INT8 Compressed:       606 KB  (74.8% Memory Reduction)
Quantization Accuracy Delta:  -0.15 pts (Near-Lossless)
Peak Operational RAM:         < 850 KB (Microcontroller Compatible)
Embedded Latency:             180–220 ms per slice on ARM Cortex
```

- **Quantization Standard:** Full-integer INT8 conversion via TensorFlow Lite Micro (`tflite.OpsSet.TFLITE_BUILTINS_INT8`).
- **Edge Targets:** Embedded medical tablets, Raspberry Pi, ARM Cortex-M processors, and offline diagnostic kiosks.

---

## 🔬 Interpretability & Diagnostic Saliency (Grad-CAM)

To provide transparency and clinical accountability, **Grad-CAM** saliency maps confirm anatomical attention:
- **Glioma Cases:** Attention concentrates on diffuse intra-axial lesion boundaries.
- **Meningioma Cases:** Activations highlight extra-axial masses attached to the dura mater.
- **Pituitary Cases:** Saliency isolates the central sella turcica region.

---

## 📓 Four-Stage Notebook Pipeline Architecture

1. **`qafnet-nb1.ipynb` — Data Harmonization & Deduplication Audit:**
   - Pairwise hash calibration ($\tau = 0.9995$).
   - Extraction of duplicate-free cohorts and patient-stratified splits.
2. **`qafnet-nb2.ipynb` — Baseline & Contamination Benchmarking:**
   - Evaluates centralized upper bounds and single-site lower bounds.
   - Quantifies the artificial accuracy inflation caused by duplicate leakage.
3. **`qafnet-nb3-v2.ipynb` — Federated Training Engine:**
   - Executes multi-round federated optimization across CHENG, PMRAM, SARTAJ, and BR35H.
   - Evaluates FedAvg, FedProx, FedBN, and QAF-Net across random seeds.
4. **`qafnet-nb4.ipynb` — TinyML Compression & Visual Saliency:**
   - Exports 606 KB TFLite INT8 binaries.
   - Computes Grad-CAM heatmaps and generates publication figures.

---

## 📂 Repository Structure & Model Checkpoints

```
QAF-Net-Brain-Tumor-MRI-Classification/
├── archive/                                      # Experimental models & metrics
│   ├── cheng_0/                                  # Seed 0 models (CHENG held-out fold)
│   │   ├── qafnet_CHENG_s0_mobilenet.tflite      # 606 KB INT8 Quantized model
│   │   ├── qafnet_CHENG_s0_mobilenet.weights.h5  # Trained Keras FP32 weights
│   │   ├── qafnet_CHENG_s0_mobilenet_history.json# Training loss/accuracy trajectory
│   │   ├── fedavg_CHENG_s0_mobilenet.tflite      # FedAvg comparative checkpoint
│   │   ├── fedbn_CHENG_s0_mobilenet.tflite       # FedBN comparative checkpoint
│   │   └── results.csv                           # Full evaluation metrics log
│   ├── cheng_1/, cheng_2/                        # Seed 1 and 2 runs
│   ├── pmram_0/, pmram_1/, pmram_2/              # PMRAM held-out fold runs
│   ├── sartaj_0/, sartaj_1/, sartaj_2/           # SARTAJ held-out fold runs
│   └── nb2_baselines/                            # Centralized & local baseline models
├── DIP.pptx                                      # 15-slide presentation slide deck
├── Group_01_Brain_Tumor_Classification...pdf     # Complete research paper PDF
├── qafnet-nb1.ipynb                              # Deduplication audit notebook
├── qafnet-nb2.ipynb                              # Baseline evaluation notebook
├── qafnet-nb3-v2.ipynb                           # Main federated learning notebook
├── qafnet-nb4.ipynb                              # TinyML export & Grad-CAM notebook
├── .gitignore                                    # Clean repository rules
├── LICENSE                                       # MIT License
└── README.md                                     # Research documentation
```

---

## 🚀 Setup & Reproduction Guide

```bash
# 1. Clone repository
git clone https://github.com/isratjahanrani/QAF-Net-Brain-Tumor-MRI-Classification.git
cd QAF-Net-Brain-Tumor-MRI-Classification

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow numpy scikit-learn matplotlib opencv-python jupyter

# 4. Launch Jupyter Notebook
jupyter notebook
```

---

## 👥 Authors & Academic Context

This research was developed as a Capstone Research Project for the **Digital Image Processing (DIP)** course in the **Department of Computer Science and Engineering** at **United International University (UIU)**, Dhaka, Bangladesh:

* **Israt Jahan Rani** (Student ID: 011201376) — *Medical Image Processing, Deduplication Audit & Domain Shift Lead* | [GitHub Profile](https://github.com/isratjahanrani)
* **Robiul Islam Tamim** (Student ID: 011212039) — *Lead AI Systems, Federated Aggregation & Quantization Lead* | [GitHub Profile](https://github.com/robiultamim)
* **Samsad Ibne Shakhawat** (Student ID: 011221407) — *Lead Author & Experimental Pipeline*
* **Most. Jaynabunnessa Rumi** (Student ID: 011221410) — *Data Analysis & Clinical Validation*

**Institutional Affiliation:** Department of Computer Science & Engineering, United International University (UIU)  
**Research Focus:** Medical Image Analysis, Privacy-Preserving Federated Learning, TinyML Edge Systems.

---

## 📄 License & Acknowledgments

- **License:** Distributed under the [MIT License](LICENSE).
- **Institutional Acknowledgment:** Sincere gratitude to the Department of CSE and laboratory instructors at **United International University (UIU)** for providing compute facilities and academic mentorship.
- **Dataset Credits:** Gratefully acknowledging the open-source medical imaging cohorts from Nanfang/Tianyou hospitals (CHENG) and the Bangladeshi PMRAM team.
