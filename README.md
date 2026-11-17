# CISC3024 AI Assignment 3

Real-matrix bilateral generalized 2DPCA for face recognition, compared with PCA, 2DPCA, and LDA on the Olivetti faces dataset.

The selected method is the grayscale special case of BiG2DQPCA:

> Mei-Xiang Zhao, Zhi-Gang Jia, Dun-Wei Gong, and Yong Zhang.  
> *Data-Driven Bilateral Generalized Two-Dimensional Quaternion Principal Component Analysis with Application to Color Face Recognition.*  
> arXiv:2306.07045, 2023. https://arxiv.org/abs/2306.07045

## Run

Use Python 3.14 (the default `python` on this machine may be 3.15, which does not have the scientific packages):

```powershell
py -3.14 -m pip install -r requirements.txt
py -3.14 experiments/run_experiments.py
py -3.14 experiments/build_report.py
```

`run_experiments.py` downloads Olivetti faces into `.cache/` on the first run, then writes `results/summary.json` and `results/figures/`.

## Layout

- `src/big2dpca.py` — Algorithm 2 of BiG2DQPCA for real matrices, plus the exact eigenvector solution used when `s = p = 2`
- `src/baselines.py` — eigenfaces PCA, Yang’s 2DPCA, and SVD linear discriminant analysis
- `src/data_utils.py` — dataset, subject-wise splits, occlusion, nearest neighbor
- `experiments/run_experiments.py` — the reported experiments
- `experiments/build_report.py` — builds the Word report from `results/summary.json`
- `AI_Assignment3_Report.docx` — the report
