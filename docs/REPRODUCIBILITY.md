# WasteAI — Reproducibility Notes

This document records the dataset split, common image pipeline, architecture-specific input preprocessing, and the environment snapshot used for the local development setup.

## 1. Dataset and split

WasteAI uses the Garbage Classification dataset:
https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

The Stage 2 manifest and split files contain 2,526 images:

| Split | Images |
|---|---:|
| Training | 1,768 |
| Validation | 379 |
| Test | 379 |
| **Total** | **2,526** |

The saved Stage 2 configuration records:

- Input size: 224 × 224 pixels, RGB.
- Random seed: 42.
- Split proportions: 70% training, 15% validation, 15% test.
- Base pipeline batch size: 32. Some architecture-specific notebook pipelines may override batch size.
- Declared augmentation operations: horizontal flip, rotation, zoom and contrast.
- Class balancing: balanced class weights.
- Class order: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

The Stage 1 manifest originally counted 2,527 valid images. One image was unavailable during Stage 2 manifest validation; the experiments and split files use the 2,526-image Stage 2 set.

The dataset itself is not distributed in this repository. Reproduction requires obtaining it independently and reviewing the dataset provider's terms.

## 2. Common image pipeline

The notebook's common `load_and_preprocess_image` function:

1. Reads the image with `tf.io.read_file`.
2. Decodes it as a three-channel RGB image.
3. Resizes it using `tf.image.resize(image, IMG_SIZE)`.
4. Converts pixel values to `float32` and divides by 255.

The common output is therefore a float image with values normalized to `[0, 1]`, before any architecture-specific transformation.

## 3. Architecture-specific preprocessing

The notebook then adapts the common `[0, 1]` tensor for the transfer-learning architectures.

| Model | Training transformation after common pipeline | Inference transformation in `app.py` |
|---|---|---|
| Custom CNN | Uses common `[0, 1]` values directly | Divide RGB pixel values by 255 |
| MobileNetV2 | Multiply by 255, then call `tf.keras.applications.mobilenet_v2.preprocess_input` | Keras MobileNetV2 `preprocess_input` on RGB values in `[0, 255]` |
| VGG16 | Multiply by 255, then call `tf.keras.applications.vgg16.preprocess_input` | Keras VGG16 `preprocess_input` on RGB values in `[0, 255]` |
| VGG19 | Multiply by 255, then call `tf.keras.applications.vgg19.preprocess_input` | Keras VGG19 `preprocess_input` on RGB values in `[0, 255]` |

**Interpretation:** The `normalization: pixel_values_divided_by_255` value in `outputs/stage2_config.json` describes the common dataset pipeline. It does not mean that MobileNetV2, VGG16 and VGG19 were trained only on `[0, 1]` values. Their notebook-specific transformations multiply by 255 before applying the matching Keras preprocessing function.

The current architecture-specific preprocessing logic in `app.py` matches the corresponding transformations shown in notebook cells 53, 78 and 103, as well as the Custom CNN's common pipeline.

## 4. Resizing parity

The training notebook uses `tf.image.resize`, whose default interpolation is bilinear. The application previously used Pillow's default `Image.resize` interpolation. Both produce the configured image dimensions, but they are not guaranteed to produce identical pixel values.

For closer training/inference parity, use TensorFlow bilinear resizing in `app.py` as shown in `patches/app_resize_patch.md`. The patch retains EXIF orientation handling in the application before resizing.

## 5. Recorded local environment

The accompanying `requirements-lock-windows-py311.txt` was copied from the project's local `python -m pip freeze` output.

| Component | Recorded version |
|---|---|
| Python | 3.11.9 |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| Streamlit | 1.67.0 |
| Altair | 5.5.0 |
| Hugging Face Hub | 2.4.0 |
| NumPy | 2.4.6 |
| Pandas | 3.1.1 |
| Pillow | 13.0.0 |
| scikit-learn | 1.9.1 |

The lock file is an environment snapshot from the developer's Windows / Python 3.11.9 environment. It is not a guaranteed cross-platform lock and should not automatically replace the Streamlit Cloud `requirements.txt`. Cloud deployment should be tested on its own Linux runtime.

To record the active environment again:

```powershell
python --version
python -m pip freeze > requirements-lock-windows-py311.txt
```

## 6. Evaluation protocol

The recorded model comparison uses the held-out Stage 2 test split. The repository includes the final model comparison CSV, per-model classification reports, confusion matrices and training histories under `outputs/`.

Recorded aggregate metrics:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| VGG19 | 91.29% | 90.43% | 89.76% | 89.93% |
| VGG16 | 88.65% | 87.58% | 88.14% | 87.69% |
| MobileNetV2 | 87.07% | 87.29% | 85.67% | 86.01% |
| Custom CNN | 73.09% | 70.27% | 72.12% | 69.62% |

These results describe the project's held-out test split. They do not establish performance on every camera, lighting condition, geography or real-world waste stream. Model softmax scores are not necessarily calibrated probabilities.

## 7. Reproduction checklist

- [ ] Obtain the dataset independently and review its terms.
- [ ] Recreate the documented Stage 2 split or use the committed split CSVs with the same image files.
- [ ] Preserve the class order from `outputs/class_mapping.json`.
- [ ] Apply the common image loading, resizing and normalization pipeline.
- [ ] Apply each model's architecture-specific preprocessing exactly as documented.
- [ ] Use the relevant notebook model definition and training procedure.
- [ ] Evaluate on the held-out Stage 2 test split.
- [ ] Compare results with `outputs/final_model_comparison.csv`.
- [ ] Record the hardware, OS, Python and dependency versions for any new run.

## 8. Scope and caveats

This document records the configuration and code paths present in the repository. It does not claim that a fresh end-to-end training run has been independently reproduced. Exact numerical results can vary with hardware, TensorFlow kernels, random operations and dependency/runtime differences.
