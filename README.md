# SHAP-Guided Grouped Regularization

This repository extends the [CRISPR-M with epigenetic features](https://github.com/lyotvincent/CRISPR-M/tree/master/test/6epigenetic/CRISPR-M) variant by introducing **SHAP-guided grouped L1 regularization**.  
It includes modifications to the original CRISPR-M codebase, a function for SHAP value computation, and three model variants:

- **L1-Grouped-Epigenetics**
- **L1-Uniform-Epigenetics** *(control model)*
- **L1-Grouped-Complete**

## Overview
The project builds on top of CRISPR-M, a multi-view deep learning model for sgRNA off-target prediction, by:
- Implementing a SHAP-based function to compute **grouped L1 regularization strengths**.
- Integrating these penalties into the three model variants listed above.
- Providing analysis notebooks for:
  - Regularization strength computation
  - Training stability comparison

## Contents
- `penalties_calculation/` — Scripts and notebooks for computing SHAP-guided group-wise L1 regularization strengths for both models.
- `shap_values/` — Precomputed SHAP values used for regularization strength calculations.
- Modified CRISPR-M core code — Includes the new SHAP computation function and changes to support the L1-grouped models.
- `stability_analysis/` — Comparison of training stability between CRISPR-M and L1-Grouped-Complete.

## Acknowledgement
This work builds upon the CRISPR-M model originally published by Sun J, Guo J, Liu J.  
Original repository: [CRISPR-M](https://github.com/lyotvincent/CRISPR-M)  

---

**License:** This repository follows the MIT license as in the original CRISPR-M project.
