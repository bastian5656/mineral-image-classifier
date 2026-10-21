# 🪨 Mineral Image Classifier

**A CNN built from scratch with PyTorch to classify mineral images — deployed as a live demo on Hugging Face Spaces.**

👉 **Live demo:** https://huggingface.co/spaces/baalboza/mineral_classifier

The model classifies photos into five minerals relevant to Chilean copper deposits: **azurite, copper, malachite, pyrite and quartz**.

---

## 📊 Result

| Metric | Value |
|---|---|
| Test accuracy (5 classes) | **85.65%** |

The confusion matrix and the full experiment log are in [`exploration.ipynb`](exploration.ipynb).

---

## 🧪 How the model was developed

The goal was to practice building a network from the ground up, so **no transfer learning** was used. The notebook documents each iteration and the reason behind it:

1. **Baseline CNN** → ~75% test accuracy, overfitting after epoch 5.
2. **Dropout** → reduced overfitting, but accuracy plateaued.
3. **Data augmentation** (flips, rotations) → overfitting disappeared, slower learning.
4. **BatchNorm + average pooling + an extra conv block** → faster, more stable convergence.
5. **Input normalization, AdamW (weight decay) and a learning-rate scheduler** → better convergence.
6. **SiLU activation and 224×224 input resolution** → final model, **85.65%** test accuracy.

### Final architecture
- Convolutional blocks: Conv2d → BatchNorm → SiLU, with pooling
- Fully connected classifier head with Dropout (0.3)
- Loss: Cross-Entropy · Optimizer: AdamW (lr = 1e-3, weight decay = 1e-4)

---

## 🗂️ Dataset

[Mineral Photos — Kaggle](https://www.kaggle.com/datasets/floriangeillon/mineral-photos). The original dataset has 15 minerals; this project uses 5 of them.

---

## 🧩 Repository structure

```
mineral-image-classifier/
├── exploration.ipynb          # Full process: data, experiments, evaluation, demo
├── models/
│   └── mineral_classifier.pth # Trained weights
├── helper_function/
│   ├── new_dataset.py         # Dataset creation and dataloaders
│   ├── helper_functions.py    # Plotting utilities
│   └── utils.py               # Model saving
├── examples/                  # Sample images
└── HF_screenshot.png          # Demo preview
```

The Gradio app (`app.py`, `model.py`, `requirements.txt`) lives in the [Hugging Face Space](https://huggingface.co/spaces/baalboza/mineral_classifier).

---

## 🖼️ Demo preview

![Hugging Face demo](HF_screenshot.png)

| Azurite | Copper | Malachite | Pyrite | Quartz |
|---|---|---|---|---|
| ![](examples/azurita.jpeg) | ![](examples/copper.jpeg) | ![](examples/malachite.jpeg) | ![](examples/pirita.jpeg) | ![](examples/quartz.jpeg) |

---

## 🛠️ Stack
Python · PyTorch · torchvision · torchmetrics · Gradio · Hugging Face Spaces · NumPy · pandas · Matplotlib

## 📞 Contact
**Bastián Albornoz** — Metallurgical Civil Engineer
[LinkedIn](https://www.linkedin.com/in/bastian-albornoz-e/) · bastian.alejandro.a@gmail.com

