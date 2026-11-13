# WasteAI — Final public QA checklist

**Live app:** https://wasteai-atchyuteswar.streamlit.app/  
**GitHub:** https://github.com/Atchyuteswar/WasteAI  
**Hugging Face:** https://huggingface.co/Atchyuteswar/wasteai-models

## A. Public access
- [ ] Open the app in a logged-out or incognito browser.
- [ ] Confirm the app loads without an application exception.
- [ ] Confirm all five navigation pages open.
- [ ] Confirm source-code and model-hosting links open correctly.

## B. Upload behavior
- [ ] Upload a JPG.
- [ ] Upload a JPEG.
- [ ] Upload a PNG.
- [ ] Upload a WebP.
- [ ] Confirm unsupported file types are rejected.
- [ ] Confirm an uploaded image preview appears.
- [ ] Confirm a corrupt image produces a readable error.

## C. Model smoke tests
Use the same image for all four model choices and record the outcome.

| Model | Loaded | Prediction shown | All 6 scores shown | No runtime error |
|---|---|---|---|---|
| Custom CNN | [ ] | [ ] | [ ] | [ ] |
| MobileNetV2 | [ ] | [ ] | [ ] | [ ] |
| VGG16 | [ ] | [ ] | [ ] | [ ] |
| VGG19 | [ ] | [ ] | [ ] | [ ] |

## D. Category coverage
Use multiple varied images per category, ideally images not in the training dataset.

| Actual category | Test 1 | Test 2 | Test 3 |
|---|---|---|---|
| Cardboard | [ ] | [ ] | [ ] |
| Glass | [ ] | [ ] | [ ] |
| Metal | [ ] | [ ] | [ ] |
| Paper | [ ] | [ ] | [ ] |
| Plastic | [ ] | [ ] | [ ] |
| Trash | [ ] | [ ] | [ ] |

Record selected model, actual image label, predicted label, top model score, cold-start/model-download time and warm prediction time.

## E. Analytics
- [ ] Model comparison shows all four models.
- [ ] Accuracy/precision/recall/F1 values agree with `outputs/final_model_comparison.csv`.
- [ ] Prediction score axis is fixed to 0–100%.
- [ ] All four confusion matrices display.
- [ ] All four accuracy/loss curves display.
- [ ] Missing optional charts show a friendly message rather than crashing.

## F. Resource and reliability tests
- [ ] Test one prediction after a fresh app restart.
- [ ] Test a second prediction with the same model to check caching.
- [ ] Test switching from a smaller model to VGG16/VGG19.
- [ ] Check Streamlit logs for memory, download or model-deserialization errors.
- [ ] Check behavior after an idle period / app wake-up.
- [ ] Record slow first-load time separately from prediction time.

## G. Privacy and responsible-use checks
- [ ] README states that the model is an educational prototype.
- [ ] README explains that softmax scores are not calibrated probabilities.
- [ ] README explains that the model does not determine exact material composition or local recyclability.
- [ ] Do not upload private or sensitive images during public testing.
- [ ] Do not expose Hugging Face write tokens or other secrets.

## Release sign-off
- [ ] All fixes committed to GitHub.
- [ ] Streamlit deployment has rebuilt from the intended commit.
- [ ] Live app tested after the latest deployment.
- [ ] README links and visual assets render on GitHub.
- [ ] Release notes prepared.
