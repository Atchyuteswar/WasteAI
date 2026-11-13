# WasteAI — Public Release Test Plan

**Live application:** https://wasteai-atchyuteswar.streamlit.app/  
**Source code:** https://github.com/Atchyuteswar/WasteAI  
**Model repository:** https://huggingface.co/Atchyuteswar/wasteai-models

## Public smoke tests

- [ ] Open the live demo in a private/incognito browser window.
- [ ] Confirm Overview loads without an application exception.
- [ ] Upload JPG, JPEG, PNG and WebP samples.
- [ ] Confirm unsupported file types are rejected.
- [ ] Confirm the uploaded image preview appears.
- [ ] Run Custom CNN prediction.
- [ ] Run MobileNetV2 prediction.
- [ ] Run VGG16 prediction.
- [ ] Run VGG19 prediction.
- [ ] Confirm all six class scores are visible.
- [ ] Confirm score charts use a fixed 0–100% horizontal axis.
- [ ] Confirm disposal guidance is readable.
- [ ] Open Model Performance and verify all four metric rows.
- [ ] Open Training Analytics and verify curves and confusion matrices.
- [ ] Open Project Details and test the GitHub/Hugging Face links.
- [ ] Repeat a prediction and check whether the selected model is cached.

## Real-world sample matrix

Use varied images that were not part of the training set.

| Category | Sample A | Sample B | Sample C |
|---|---|---|---|
| Cardboard | [ ] | [ ] | [ ] |
| Glass | [ ] | [ ] | [ ] |
| Metal | [ ] | [ ] | [ ] |
| Paper | [ ] | [ ] | [ ] |
| Plastic | [ ] | [ ] | [ ] |
| Trash | [ ] | [ ] | [ ] |

## Record each test

Record the selected model, actual image category, predicted category, top model score, first-load time, repeat prediction time, and any error or unexpected behavior.

## Interpretation and privacy cautions

- A softmax score is not necessarily a calibrated probability of correctness.
- The dataset label “trash” is a class label, not a material-composition analysis.
- Recyclability depends on material, contamination, local rules, and available facilities.
- Held-out test accuracy does not guarantee performance on new real-world images.
- Do not upload sensitive or private images to the public demo.
