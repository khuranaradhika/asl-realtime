# Applied Deep Learning Final Project

**Course:** Applied Deep Learning  
**Group Members:** Gyula Planky · Hrishikesh Pradhan · Jian Gao · Radhika Khurana

---

## Project Overview

> _[Fill in: brief description of your proposed project idea and motivation.]_

**Project Area:** `[ ] Perception` &nbsp;|&nbsp; `[ ] Behavior` &nbsp;|&nbsp; `[ ] Other Signals`  
**Data Types:** `[ ] Images` &nbsp;|&nbsp; `[ ] Video` &nbsp;|&nbsp; `[ ] RF` &nbsp;|&nbsp; `[ ] Other: ___`

---

## Repository Structure

```
applied_deep_learning_final/
│
├── data/
│   ├── raw/               # Original, unmodified datasets
│   └── processed/         # Cleaned/transformed data for modeling
│
├── src/
│   ├── dataloader.py      # PyTorch Dataset/DataLoader definitions
│   └── ...                # Other source scripts
│
├── notebooks/             # Jupyter notebooks for EDA and experiments
│
├── models/                # Saved model weights and checkpoints
│
├── docs/                  # Documentation, experiment logs, project plans
│
├── results/               # Output plots, metrics, evaluation results
│
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Setup & Environment

### 1. Clone the repo
```bash
git clone https://github.com/khuranaradhika/applied_deep_learning_final.git
cd applied_deep_learning_final
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> Please always activate your virtual environment before running any scripts or notebooks, and update `requirements.txt` when you add new packages (`pip freeze > requirements.txt`).

---

## PyTorch DataLoader

We use the [PyTorch Dataset & DataLoader API](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) for all data loading. The standard template lives in `src/dataloader.py` — create new dataloaders in the same style to keep things consistent across the team.

---

## Jupyter Notebook Guidelines

- **Clear outputs before committing** to avoid merge conflicts and repo bloat.
- From the terminal: `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
- Keep notebooks in `notebooks/` — don't scatter them across the repo.

---

## Git Workflow

- `main` — stable, working code only
- Feature branches: `feature/your-name-description` (e.g., `feature/radhika-dataloader`)
- Open a PR and get at least one review before merging into `main`
- Write meaningful commit messages: `Add ResNet baseline` not `update stuff`
- Pull before you push — avoid unnecessary conflicts

---

## Experiments & Planning

> _[Document planned experiments, ablations, and steps here — or link to a doc in `docs/`.]_

- [ ] EDA on raw data
- [ ] Baseline model
- [ ] Experiment 1: _____
- [ ] Experiment 2: _____

---

## Team Members and Contributions

| Name | GitHub |
|------|--------|
| Radhika Khurana | [@khuranaradhika](https://github.com/khuranaradhika) | Created github (sample text area)
| Gyula Planky | — |
| Hrishikesh Pradhan | — |
| Jian Gao | — |
