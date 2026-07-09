# Recruiter Brief: Industrial Control System Cyber Attack Detection

## One-minute summary

This academic project explores machine-learning detection of cyber attacks and natural disturbances in industrial power-control data. It combines notebooks, trained model artifacts, sample datasets, and a Flask web interface for uploading CSV measurements and returning classifications. It demonstrates applied ML, cybersecurity research, data preprocessing, model evaluation, and user-facing tooling around critical-infrastructure security.

## What it demonstrates

- Applied machine learning with scikit-learn models including Random Forest, neural networks, KNN, decision trees, logistic regression, AdaBoost, gradient boosting, and extra trees.
- Cybersecurity domain work around industrial control systems, power-system measurements, attack/disruption classification, and evaluation against known public research datasets.
- Data engineering and preprocessing with CSV/ARFF datasets, feature scaling, feature selection, serialized model pipelines, and reproducible notebooks.
- Practical productization through a Flask upload interface that routes user data into the prediction workflow.

## Architecture at a glance

```text
CSV power-system measurements
  -> preprocessing and feature cleanup
  -> router/model selection
  -> serialized scikit-learn classifiers
  -> predicted class labels
  -> Flask results page
```

Important files:

- `ReadME.md` - project overview and dataset references.
- `Industrial_power_control_system_cyber_attacks_detetection_ui.py` - core preprocessing and prediction flow.
- `WebPage/app.py` - Flask interface for file upload and results display.
- `Class/` - binary, triple, and multiclass dataset variants.
- `P/` and numbered model folders - serialized model artifacts used by the prediction flow.
- `*.ipynb` - experiment notebooks and model-building work.
- `Articles/` - supporting research references.

## How to verify locally

```bash
pip install -r requirements.txt
python WebPage/app.py
```

Then upload a CSV matching the dataset format described in `ReadME.md`. For experiment reproduction, open the notebooks and rerun the relevant preprocessing/modeling pipeline with the included datasets.

## Role positioning

Best aligned roles:

- Machine Learning Engineer
- Cybersecurity Researcher
- Data Scientist
- Python Developer
- Software Engineer, security-adjacent

Suggested portfolio phrasing:

> Built an ML-based industrial-control intrusion-detection prototype using public power-system datasets, scikit-learn classifiers, serialized model pipelines, and a Flask upload interface for classification.

## Responsible claims

This is an educational and research project. It should not be presented as production-ready protection for critical infrastructure without additional validation, monitoring, operational hardening, and domain-specific review.
