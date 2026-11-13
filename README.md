# ♻️ WasteAI — Deep Learning-Based Waste Classification

[![Open Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wasteai-atchyuteswar.streamlit.app/)

**Live demo:** https://wasteai-atchyuteswar.streamlit.app/

WasteAI is an academic research prototype that classifies waste images into six categories using deep learning. It compares a custom CNN with MobileNetV2, VGG16 and VGG19, and provides a Streamlit interface for image classification and experiment analysis.

> **Deployment status:** Source code and experiment artifacts are in this repository. To make the public demo fully functional, publish the four model weights on Hugging Face and configure the Streamlit app as described below.

## Project results

Recorded results from the project's held-out test split:

| Rank | Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---:|---|---:|---:|---:|---:|
| 1 | **VGG19** | **91.29%** | 90.43% | 89.76% | 89.93% |
| 2 | VGG16 | 88.65% | 87.58% | 88.14% | 87.69% |
| 3 | MobileNetV2 | 87.07% | 87.29% | 85.67% | 86.01% |
| 4 | Custom CNN | 73.09% | 70.27% | 72.12% | 69.62% |

The Stage 2 split used 1,768 training, 379 validation and 379 test images (2,526 total). The earlier Stage 1 report counted 2,527 valid images; one image was unavailable during Stage 2 manifest validation. The experiments used the saved Stage 2 split.

These figures are experimental results, not a guarantee of performance on unseen real-world images. Per-class reports, training curves and confusion matrices are available in `outputs/`.

## Classes

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Features

- Upload JPG, JPEG, PNG or WebP images.
- Select Custom CNN, MobileNetV2, VGG16 or VGG19.
- View the predicted category and class-score distribution.
- View general disposal guidance.
- Compare recorded model metrics.
- Explore training curves and confusion matrices.

## Repository layout

```text
WasteAI/
├── app.py
├── requirements.txt
├── README.md
├── MODEL_HOSTING.md
├── .gitignore
├── outputs/                 # Metrics, charts, reports and class mapping
├── Waste_Classification_Stage_1.ipynb
└── models/                   # Local-only; model weights are not committed
```

The source repository intentionally excludes the original dataset and `.keras` model weights. The model files are hosted separately to avoid GitHub's per-file size restrictions.

## Run locally

Recommended: Python 3.11.

```bash
git clone https://github.com/Atchyuteswar/WasteAI.git
cd WasteAI
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and launch:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

For local inference, put the four selected `.keras` files in `models/`. Alternatively, configure `HF_MODEL_REPO` to download them from Hugging Face. Details: [MODEL_HOSTING.md](MODEL_HOSTING.md).

## Deploy a public demo

1. Upload the four model files to a public Hugging Face model repository. Use the exact filenames in [MODEL_HOSTING.md](MODEL_HOSTING.md).
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app from this repository, branch `main`, main file `app.py`.
3. Select Python 3.11 in the deployment settings if the option is shown.
4. In the Streamlit app's **Settings → Secrets**, add:

   ```toml
   HF_MODEL_REPO = "YOUR_HF_USERNAME/wasteai-models"
   ```

5. Save secrets and reboot the app. Test all four model choices.

For public Hugging Face model repos, a token is not required. If using a private model repo, add `HF_TOKEN` as a Streamlit secret; never place tokens in source code or Git.

## Dataset

The project uses the [Garbage Classification dataset on Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification). Review the dataset's terms and license before reuse or redistribution.

## Limitations

- This is an educational research prototype, not a certified waste-sorting or municipal decision system.
- A class prediction does not determine exact composition, plastic resin, contamination or local recyclability.
- Disposal guidance is general; municipal rules vary.
- Softmax scores are not necessarily calibrated probabilities.
- Test accuracy depends on the selected dataset split and may not generalize to real-world images.

## License

No open-source license has been selected yet. Public visibility does not automatically grant permission to reuse or redistribute this project. Add a license file if you intend to grant reuse rights.
