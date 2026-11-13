# Model hosting with Hugging Face Hub

The Git repository excludes large `.keras` files. Host the four inference models in a Hugging Face **public model repository** so the public Streamlit app can download only the model a visitor selects.

## 1. Create the model repository

1. Sign in at https://huggingface.co/.
2. Create a new **Model** repository, for example `wasteai-models`.
3. Set it to **Public** for token-free downloads from the public demo.
4. Upload these four files from `F:\Waste_Classification\models\`:

| Filename | Architecture |
|---|---|
| `cnn_model.keras` | Custom CNN |
| `mobilenetv2_model.keras` | MobileNetV2 |
| `vgg16_model.keras` | VGG16 |
| `vgg19_model.keras` | VGG19 |

Only these four inference files are needed. The `*_best.keras`, `*_head_best.keras` and `*_finetuned_best.keras` checkpoints are optional research artifacts and are not used by the app.

## 2. Upload from your computer (CLI option)

Install the Hub client:

```powershell
pip install huggingface_hub
```

Log in using a Hugging Face **write** token interactively (do not paste it into source code):

```powershell
hf auth login
```

Upload each file (replace the repository ID):

```powershell
hf upload YOUR_HF_USERNAME/wasteai-models F:\Waste_Classification\models\cnn_model.keras cnn_model.keras --repo-type model
hf upload YOUR_HF_USERNAME/wasteai-models F:\Waste_Classification\models\mobilenetv2_model.keras mobilenetv2_model.keras --repo-type model
hf upload YOUR_HF_USERNAME/wasteai-models F:\Waste_Classification\models\vgg16_model.keras vgg16_model.keras --repo-type model
hf upload YOUR_HF_USERNAME/wasteai-models F:\Waste_Classification\models\vgg19_model.keras vgg19_model.keras --repo-type model
```

You can also upload through the Hugging Face website's Files and versions tab.

## 3. Configure Streamlit Community Cloud

After deploying `app.py` from this GitHub repository, open the app's **Settings → Secrets** and add:

```toml
HF_MODEL_REPO = "YOUR_HF_USERNAME/wasteai-models"
```

Replace the example with your actual Hugging Face account and repository. A public model repository can be downloaded without a token.

For a private repository only, add a read token as a Streamlit secret:

```toml
HF_MODEL_REPO = "YOUR_HF_USERNAME/wasteai-models"
HF_TOKEN = "hf_your_read_token"
```

Never commit `HF_TOKEN` to GitHub or include it in a screenshot.

## 4. Local development

The app checks the local `models/` folder first. If the selected file is absent, it downloads from the configured Hugging Face repo.

PowerShell example:

```powershell
$env:HF_MODEL_REPO = "YOUR_HF_USERNAME/wasteai-models"
streamlit run app.py
```

## 5. Resource considerations

- Check each model's size before uploading; Hugging Face storage and bandwidth are subject to current account quotas and policies.
- Streamlit Community Cloud has finite RAM and CPU. The app caches one model at a time to reduce memory use, but VGG models may still exceed free-tier resources depending on model size and platform limits.
- If deployment runs out of memory, use a host with more RAM or deploy model inference separately behind an API.

## 6. Preprocessing must match training

The application currently assumes:

- Custom CNN: RGB pixels scaled to `[0, 1]`.
- MobileNetV2: Keras MobileNetV2 `preprocess_input` on RGB pixels in `[0, 255]`.
- VGG16: Keras VGG16 `preprocess_input` on RGB pixels in `[0, 255]`.
- VGG19: Keras VGG19 `preprocess_input` on RGB pixels in `[0, 255]`.

These settings reflect the Colab pipeline described in the project work. Confirm them against the notebook before treating deployed predictions as validated.
