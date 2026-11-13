"""
WasteAI — Deep Learning Waste Classification
A professional, responsive Streamlit application built with native Streamlit components.

Run:
    streamlit run app.py

Expected project layout:
    Waste_Classification/
      app.py
      .streamlit/
        config.toml
      models/
        cnn_model.keras
        mobilenetv2_model.keras
        vgg16_model.keras
        vgg19_model.keras
      outputs/
        class_mapping.json
        final_model_comparison.csv
        *_confusion_matrix.png
        *_accuracy.png
        *_loss.png
        final_*comparison*.png
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WasteAI | Waste intelligence",
    page_icon=":material/recycling:",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


def get_setting(name: str, default: str = "") -> str:
    """Read Streamlit Cloud secrets, falling back to environment variables."""
    try:
        value = st.secrets.get(name, None)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


HF_MODEL_REPO = get_setting("HF_MODEL_REPO")
HF_TOKEN = get_setting("HF_TOKEN") or None

DEFAULT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "VGG19": {
        "filename": "vgg19_model.keras",
        "prefix": "vgg19",
        "preprocessing": "vgg19",
        "test_accuracy": 91.29,
        "size_mb": 150.03,
        "badge_color": "green",
        "role": "Top accuracy",
        "description": "Deep 19-layer VGG architecture with ImageNet transfer learning.",
    },
    "VGG16": {
        "filename": "vgg16_model.keras",
        "prefix": "vgg16",
        "preprocessing": "vgg16",
        "test_accuracy": 88.65,
        "size_mb": 111.76,
        "badge_color": "blue",
        "role": "Balanced depth",
        "description": "16-layer VGG architecture with ImageNet transfer learning.",
    },
    "MobileNetV2": {
        "filename": "mobilenetv2_model.keras",
        "prefix": "mobilenetv2",
        "preprocessing": "mobilenet_v2",
        "test_accuracy": 87.07,
        "size_mb": 22.60,
        "badge_color": "violet",
        "role": "Lightweight transfer",
        "description": "Efficient depthwise-separable CNN architecture for compact deployment.",
    },
    "Custom CNN": {
        "filename": "cnn_model.keras",
        "prefix": "cnn",
        "preprocessing": "rescale",
        "test_accuracy": 73.09,
        "size_mb": 4.92,
        "badge_color": "orange",
        "role": "Baseline from scratch",
        "description": "Custom four-block convolutional neural network trained from scratch.",
    },
}

# IMPORTANT:
# These defaults assume the transfer-learning models were trained using the
# corresponding Keras preprocess_input function on RGB pixel values in [0,255],
# and the custom CNN was trained on pixels scaled to [0,1].
#
# If your Colab training pipeline used ImageDataGenerator(rescale=1./255)
# for ALL models, change the preprocessing values for VGG19, VGG16 and
# MobileNetV2 to "rescale". Inference preprocessing must exactly match training.

DISPOSAL_GUIDE: dict[str, tuple[str, str, str]] = {
    "cardboard": (
        "Usually recyclable",
        "green",
        "Keep cardboard clean and dry. Flatten boxes and remove excess tape where practical. "
        "Check your local collection rules.",
    ),
    "glass": (
        "Often recyclable",
        "green",
        "Empty and rinse containers. Accepted glass types vary by local recycling service. "
        "Broken glass may need separate handling.",
    ),
    "metal": (
        "Often recyclable",
        "green",
        "Empty and rinse cans or containers. Check local rules for accepted metal packaging.",
    ),
    "paper": (
        "Usually recyclable",
        "green",
        "Keep paper clean and dry. Greasy or food-soiled paper may not be accepted with ordinary paper.",
    ),
    "plastic": (
        "Depends on resin type and local facilities",
        "orange",
        "Recycling depends on resin type, item shape, contamination, and local facilities. "
        "Check the packaging symbol and local collection guidance.",
    ),
    "trash": (
        "General-waste dataset category",
        "gray",
        "The model's 'trash' label is a dataset category, not a definitive material analysis. "
        "Check whether the item can be reused, separated, recycled, or composted.",
    ),
}

CATEGORY_ICONS: dict[str, str] = {
    "cardboard": ":material/inventory_2:",
    "glass": ":material/wine_bar:",
    "metal": ":material/hardware:",
    "paper": ":material/description:",
    "plastic": ":material/water_bottle:",
    "trash": ":material/delete_outline:",
}

PAGE_OPTIONS = [
    "Overview",
    "Waste Classifier",
    "Model Performance",
    "Training Analytics",
    "Project Details",
]

PAGE_LABELS: dict[str, str] = {
    "Overview": ":material/dashboard: Overview",
    "Waste Classifier": ":material/photo_camera: Waste classifier",
    "Model Performance": ":material/leaderboard: Model performance",
    "Training Analytics": ":material/monitoring: Training analytics",
    "Project Details": ":material/info: Project details",
}


# ============================================================
# 2. DATA AND MODEL HELPERS
# ============================================================

@st.cache_data
def load_classes() -> list[str]:
    """Read the class-index mapping produced by the Colab pipeline."""
    mapping_file = OUTPUT_DIR / "class_mapping.json"

    if not mapping_file.exists():
        return DEFAULT_CLASSES

    try:
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))

        # Format A: {"cardboard": 0, "glass": 1, ...}
        if isinstance(mapping, dict) and mapping:
            if all(isinstance(value, (int, float)) for value in mapping.values()):
                ordered = sorted(mapping.items(), key=lambda item: item[1])
                return [str(name) for name, _ in ordered]

            # Format B: {"0": "cardboard", "1": "glass", ...}
            if all(str(key).isdigit() for key in mapping):
                ordered = sorted(mapping.items(), key=lambda item: int(item[0]))
                return [str(value) for _, value in ordered]

    except Exception:
        pass

    return DEFAULT_CLASSES


CLASSES = load_classes()


@st.cache_resource(show_spinner="Loading selected model...", max_entries=1)
def load_keras_model(model_name: str):
    """Load one selected model from local disk or Hugging Face Hub.

    max_entries=1 limits the number of cached model objects to reduce RAM use
    on free hosting tiers. Local files take precedence during development.
    """
    filename = MODEL_CONFIG[model_name]["filename"]
    local_path = MODEL_DIR / filename

    if local_path.is_file():
        model_path = str(local_path)
    else:
        if not HF_MODEL_REPO:
            raise FileNotFoundError(
                f"{filename} was not found in the local models/ folder. "
                "For cloud deployment, add HF_MODEL_REPO to Streamlit Secrets. "
                "See MODEL_HOSTING.md."
            )
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=filename,
            repo_type="model",
            token=HF_TOKEN,
        )

    return tf.keras.models.load_model(model_path, compile=False)


def get_model_input_size(model: Any) -> tuple[int, int]:
    """Use the model's declared spatial input dimensions where available."""
    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    height = shape[1] if len(shape) >= 4 else 224
    width = shape[2] if len(shape) >= 4 else 224

    if height is None or width is None:
        return 224, 224

    return int(height), int(width)


def preprocess_for_model(image: Image.Image, model_name: str, model: Any) -> np.ndarray:
    """Preprocess one RGB image using the training pipeline's normalization mode."""
    height, width = get_model_input_size(model)
    image = ImageOps.exif_transpose(image).convert("RGB").resize((width, height))
    pixels = np.asarray(image, dtype=np.float32)

    mode = MODEL_CONFIG[model_name]["preprocessing"]

    if mode == "rescale":
        pixels = pixels / 255.0
    elif mode == "mobilenet_v2":
        pixels = tf.keras.applications.mobilenet_v2.preprocess_input(pixels)
    elif mode == "vgg16":
        pixels = tf.keras.applications.vgg16.preprocess_input(pixels)
    elif mode == "vgg19":
        pixels = tf.keras.applications.vgg19.preprocess_input(pixels)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    return np.expand_dims(pixels, axis=0)


def predict_image(image: Image.Image, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a single PIL image and return normalized class scores and ranking."""
    model = load_keras_model(model_name)
    input_batch = preprocess_for_model(image, model_name, model)

    raw_output = model.predict(input_batch, verbose=0)
    scores = np.asarray(raw_output).squeeze()

    if scores.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional class scores, but model returned shape {scores.shape}."
        )

    if len(scores) != len(CLASSES):
        raise ValueError(
            f"Model output has {len(scores)} values, but class mapping has "
            f"{len(CLASSES)} classes. Check outputs/class_mapping.json."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned NaN or infinite scores.")

    # If model outputs logits rather than probabilities, convert to softmax.
    if np.any(scores < 0) or not np.isclose(float(np.sum(scores)), 1.0, atol=0.03):
        exp_scores = np.exp(scores - np.max(scores))
        scores = exp_scores / np.sum(exp_scores)

    ranking = np.argsort(scores)[::-1]
    return scores, ranking


@st.cache_data
def load_comparison_data() -> pd.DataFrame | None:
    """Load the held-out test comparison table if present."""
    path = OUTPUT_DIR / "final_model_comparison.csv"
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_training_history(prefix: str) -> pd.DataFrame | None:
    """Load per-epoch training history CSV for a given model prefix."""
    path = OUTPUT_DIR / f"{prefix}_training_history.csv"
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_classification_report(prefix: str) -> str | None:
    """Load the saved text classification report for a given model prefix."""
    path = OUTPUT_DIR / f"{prefix}_classification_report.txt"
    if not path.exists():
        return None

    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def show_output_image(filename: str, title: str, alt_text: str) -> None:
    """Render an output plot inside a bordered card with an accessible name."""
    path = OUTPUT_DIR / filename
    with st.container(border=True, height="stretch"):
        st.markdown(f"**{title}**")
        if path.exists():
            st.image(str(path), alt=alt_text, width="stretch")
        else:
            st.caption(f"Visualization not available: `outputs/{filename}`")


def navigate_to(target_page: str) -> None:
    """Callback to switch workspace pages programmatically."""
    st.session_state.workspace_page = target_page


# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================

st.session_state.setdefault("workspace_page", "Overview")

with st.sidebar:
    st.subheader(":material/recycling: WasteAI")
    st.caption("Deep learning waste intelligence platform")

    page = (
        st.radio(
            "Workspace",
            PAGE_OPTIONS,
            format_func=lambda option: PAGE_LABELS.get(option, option),
            key="workspace_page",
            bind="query-params",
        )
        or "Overview"
    )

    st.space("small")
    with st.container(border=True):
        st.caption("Project snapshot")
        with st.container(horizontal=True):
            st.badge(f"{len(CLASSES)} classes", icon=":material/category:", color="green")
            st.badge("4 CNN models", icon=":material/neurology:", color="blue")
        st.metric(
            "Best test accuracy",
            "91.29%",
            "VGG19 transfer learning",
            delta_arrow="off",
            delta_color="green",
        )

    st.caption("Academic research prototype · TensorFlow & Keras")


# ============================================================
# 4. OVERVIEW PAGE
# ============================================================

if page == "Overview":
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(
                "AI-powered material recognition",
                icon=":material/eco:",
                color="green",
            )
            st.badge(
                "6 material classes",
                icon=":material/dataset:",
                color="blue",
            )
            st.badge(
                "4 CNN architectures",
                icon=":material/memory:",
                color="violet",
            )

        st.title("Smarter waste recognition for better-informed sorting")
        st.markdown(
            "An image-based waste classification research prototype that benchmarks "
            "**Custom CNN**, **MobileNetV2**, **VGG16**, and **VGG19** across six "
            "common household and municipal material streams."
        )

        with st.container(horizontal=True):
            st.button(
                "Classify a waste image",
                icon=":material/photo_camera:",
                type="primary",
                on_click=navigate_to,
                args=("Waste Classifier",),
            )
            st.button(
                "Compare model benchmarks",
                icon=":material/leaderboard:",
                on_click=navigate_to,
                args=("Model Performance",),
            )
            st.button(
                "Inspect training curves",
                icon=":material/monitoring:",
                on_click=navigate_to,
                args=("Training Analytics",),
            )

    with st.grid(4, min_column_width=200, gap="small"):
        st.metric(
            "Material classes",
            f"{len(CLASSES):02d}",
            "Cardboard, glass, metal, paper, plastic, trash",
            icon=":material/category:",
            delta_arrow="off",
            delta_color="off",
            border=True,
            height="stretch",
        )
        st.metric(
            "Models evaluated",
            "04",
            "CNN, MobileNetV2, VGG16, VGG19",
            icon=":material/neurology:",
            delta_arrow="off",
            delta_color="off",
            border=True,
            height="stretch",
        )
        st.metric(
            "Best test accuracy",
            "91.29%",
            "+18.20% vs. custom CNN",
            icon=":material/emoji_events:",
            border=True,
            height="stretch",
        )
        st.metric(
            "Compact deployment option",
            "22.6 MB",
            "MobileNetV2 (87.07% accuracy)",
            icon=":material/bolt:",
            delta_arrow="off",
            delta_color="blue",
            border=True,
            height="stretch",
        )

    st.subheader("Explore the workspace", icon=":material/explore:")

    with st.grid(2, min_column_width=300, gap="small"):
        with st.container(border=True, height="stretch"):
            st.badge("01 · Image inference", icon=":material/photo_camera:", color="green")
            st.subheader("Identify a waste category")
            st.markdown(
                "Upload a single waste item photo, run inference with any of the four "
                "trained neural networks, and inspect class probability distributions "
                "alongside general disposal guidance."
            )
            st.button(
                "Open waste classifier",
                icon=":material/arrow_forward:",
                key="overview_goto_classifier",
                on_click=navigate_to,
                args=("Waste Classifier",),
            )

        with st.container(border=True, height="stretch"):
            st.badge("02 · Experiment analysis", icon=":material/science:", color="blue")
            st.subheader("Compare model performance")
            st.markdown(
                "Explore held-out test accuracy, macro precision, recall, F1 score, "
                "model storage footprint, per-epoch learning curves, and confusion matrices."
            )
            st.button(
                "Open performance benchmarks",
                icon=":material/arrow_forward:",
                key="overview_goto_performance",
                on_click=navigate_to,
                args=("Model Performance",),
            )

    st.subheader("Supported material categories", icon=":material/recycling:")
    with st.grid(6, min_column_width=150, gap="small"):
        for category in CLASSES:
            guide_status, badge_color, _ = DISPOSAL_GUIDE.get(
                category, ("Check local rules", "gray", "")
            )
            icon_name = CATEGORY_ICONS.get(category, ":material/recycling:")
            with st.container(border=True, height="stretch"):
                st.markdown(f"**{icon_name} {category.title()}**")
                st.badge(guide_status, color=badge_color)

    st.info(
        "The reported model metrics come from held-out test evaluations on the Garbage Classification "
        "dataset. Real-world accuracy depends on lighting, object occlusion, and background clutter.",
        title="Evaluation scope",
        icon=":material/info:",
    )


# ============================================================
# 5. WASTE CLASSIFIER PAGE
# ============================================================

elif page == "Waste Classifier":
    st.badge("Image inference", icon=":material/photo_camera:", color="green")
    st.title("Classify a waste item")
    st.markdown(
        "Upload a clear, well-lit image of a single waste item to generate a category "
        "prediction and inspect model output scores."
    )

    upload_col, preview_col = st.columns(2, gap="medium")

    with upload_col:
        with st.container(border=True, height="stretch"):
            st.subheader("1. Configure inference", icon=":material/tune:")

            uploaded_file = st.file_uploader(
                "Upload waste image",
                type=["jpg", "jpeg", "png", "webp"],
                help="Use a clear, well-lit image with the main waste item centered.",
            )

            selected_model = st.selectbox(
                "Select classification model",
                list(MODEL_CONFIG.keys()),
                index=0,
                key="classifier_model_select",
            )

            model_meta = MODEL_CONFIG[selected_model]
            with st.container(horizontal=True):
                st.badge(
                    model_meta["role"],
                    icon=":material/verified:",
                    color=model_meta["badge_color"],
                )
                st.badge(
                    f"Test accuracy: {model_meta['test_accuracy']:.2f}%",
                    icon=":material/analytics:",
                    color="gray",
                )
                st.badge(
                    f"Size: {model_meta['size_mb']:.1f} MB",
                    icon=":material/hard_drive:",
                    color="gray",
                )

            st.caption(model_meta["description"])

            classify_button = st.button(
                "Analyze waste image",
                icon=":material/auto_awesome:",
                type="primary",
                width="stretch",
                disabled=uploaded_file is None,
            )

    uploaded_image: Image.Image | None = None

    with preview_col:
        with st.container(border=True, height="stretch"):
            st.subheader("2. Image preview", icon=":material/image:")

            if uploaded_file is not None:
                try:
                    uploaded_image = Image.open(uploaded_file)
                    uploaded_image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
                    width_px, height_px = uploaded_image.size
                    st.image(
                        uploaded_image,
                        caption=f"{uploaded_file.name} ({width_px} × {height_px} px)",
                        alt=f"Uploaded waste item photo ({uploaded_file.name}) for classification",
                        width="stretch",
                    )
                except Exception:
                    st.error(
                        "The uploaded file could not be opened as an image. Please try another JPG, PNG, or WebP file.",
                        title="Invalid image file",
                        icon=":material/error:",
                    )
            else:
                st.info(
                    "Upload a JPG, PNG, or WebP image on the left to preview the item before running classification.",
                    title="No image selected",
                    icon=":material/upload_file:",
                )
                st.caption(
                    "Tip: Single-item photos against a simple background produce the most reliable predictions."
                )

    if classify_button and uploaded_image is not None:
        try:
            started = time.perf_counter()

            with st.spinner(f"Analyzing image with {selected_model}..."):
                scores, ranking = predict_image(uploaded_image, selected_model)

            elapsed = time.perf_counter() - started
            best_index = int(ranking[0])
            runner_up_index = int(ranking[1]) if len(ranking) > 1 else best_index
            predicted_class = CLASSES[best_index]
            runner_up_class = CLASSES[runner_up_index]
            top_score = float(scores[best_index])
            runner_up_score = float(scores[runner_up_index])

            guide_title, guide_color, guide_text = DISPOSAL_GUIDE.get(
                predicted_class,
                ("Check local guidance", "gray", "Follow local municipal waste collection guidance."),
            )
            pred_icon = CATEGORY_ICONS.get(predicted_class, ":material/recycling:")

            st.subheader("Classification result", icon=":material/check_circle:")

            with st.grid(3, min_column_width=220, gap="small"):
                st.metric(
                    "Predicted material",
                    predicted_class.title(),
                    f"Top score: {top_score * 100:.2f}%",
                    icon=pred_icon,
                    delta_arrow="off",
                    delta_color="green" if top_score >= 0.80 else "orange",
                    border=True,
                    height="stretch",
                )
                st.metric(
                    "Second-highest class",
                    runner_up_class.title(),
                    f"Score: {runner_up_score * 100:.2f}%",
                    icon=":material/format_list_numbered:",
                    delta_arrow="off",
                    delta_color="off",
                    border=True,
                    height="stretch",
                )
                st.metric(
                    "Inference latency",
                    f"{elapsed:.2f} s",
                    f"Architecture: {selected_model}",
                    icon=":material/timer:",
                    delta_arrow="off",
                    delta_color="off",
                    border=True,
                    height="stretch",
                )

            with st.container(border=True):
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown("**Disposal & handling guidance**")
                    st.badge(guide_title, icon=":material/recycling:", color=guide_color)
                st.markdown(guide_text)

            if top_score < 0.60:
                st.warning(
                    "The top model score is below 60%. Treat this prediction as uncertain and verify the material manually.",
                    title="Low prediction confidence",
                    icon=":material/warning:",
                )
            elif top_score < 0.80:
                st.info(
                    "The model has a moderate top score (60%–80%). Check the runner-up category and verify the material before disposal.",
                    title="Moderate prediction confidence",
                    icon=":material/info:",
                )

            with st.container(border=True):
                st.subheader("Score distribution across all classes", icon=":material/bar_chart:")

                score_table = pd.DataFrame(
                    {
                        "Material": [CLASSES[index].title() for index in ranking],
                        "Score (%)": [float(scores[index]) * 100 for index in ranking],
                    }
                )

                score_chart = (
                    alt.Chart(score_table)
                    .mark_bar(cornerRadiusEnd=5, height=24)
                    .encode(
                        x=alt.X(
                            "Score (%):Q",
                            scale=alt.Scale(domain=[0, 100], clamp=True),
                            axis=alt.Axis(
                                title="Model output score (%)",
                                values=[0, 20, 40, 60, 80, 100],
                                labelExpr="datum.value + '%'",
                            ),
                        ),
                        y=alt.Y("Material:N", sort="-x", axis=alt.Axis(title=None)),
                        tooltip=[
                            alt.Tooltip("Material:N", title="Class"),
                            alt.Tooltip("Score (%):Q", title="Score (%)", format=".2f"),
                        ],
                    )
                    .properties(height=250)
                )

                st.altair_chart(
                    score_chart,
                    alt=f"Horizontal bar chart of model output scores led by {predicted_class.title()} at {top_score * 100:.2f}%",
                )

                with st.expander("View detailed prediction scores", icon=":material/table_chart:"):
                    st.dataframe(
                        score_table,
                        hide_index=True,
                        column_config={
                            "Material": st.column_config.TextColumn("Material class"),
                            "Score (%)": st.column_config.ProgressColumn(
                                "Model output score",
                                format="%.2f%%",
                                min_value=0,
                                max_value=100,
                            ),
                        },
                        alt="Table of model output scores for all six waste categories",
                    )

                st.caption(
                    "The displayed score is the model's softmax output score and is not necessarily a calibrated probability of correctness."
                )

        except Exception as error:
            st.error(
                "The selected model could not complete classification.",
                title="Inference error",
                icon=":material/error:",
            )
            st.exception(error)
            with st.container(border=True):
                st.markdown(
                    """
                    **Troubleshooting checks**
                    - Confirm the selected `.keras` file exists in `models/` or `HF_MODEL_REPO` is configured.
                    - Confirm local TensorFlow/Keras versions are compatible with the saved model.
                    - Confirm `outputs/class_mapping.json` contains six classes.
                    - Confirm inference preprocessing matches the Colab training pipeline.
                    """
                )


# ============================================================
# 6. MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":
    st.badge("Experiment results", icon=":material/leaderboard:", color="green")
    st.title("Model performance comparison")
    st.markdown(
        "Review the four convolutional architectures evaluated on the same held-out test split."
    )

    comparison_df = load_comparison_data()

    if comparison_df is None:
        st.warning(
            "Copy the Stage 7 comparison CSV (`final_model_comparison.csv`) into `outputs/` to view the benchmark table.",
            title="Comparison data missing",
            icon=":material/warning:",
        )
    else:
        with st.grid(4, min_column_width=190, gap="small"):
            st.metric(
                "Top-ranked architecture",
                "VGG19",
                "Rank #1 overall",
                icon=":material/emoji_events:",
                delta_arrow="off",
                delta_color="green",
                border=True,
                height="stretch",
            )
            st.metric(
                "Highest test accuracy",
                "91.29%",
                "VGG19 on held-out test set",
                icon=":material/check_circle:",
                delta_arrow="off",
                delta_color="green",
                border=True,
                height="stretch",
            )
            st.metric(
                "Highest macro F1 score",
                "89.93%",
                "Precision 90.43% · Recall 89.76%",
                icon=":material/balance:",
                delta_arrow="off",
                delta_color="blue",
                border=True,
                height="stretch",
            )
            st.metric(
                "Best efficiency trade-off",
                "MobileNetV2",
                "87.07% accuracy at 22.6 MB",
                icon=":material/speed:",
                delta_arrow="off",
                delta_color="violet",
                border=True,
                height="stretch",
            )

        display_df = comparison_df.copy()
        metric_candidates = ["Accuracy", "Precision", "Recall", "F1 Score"]
        available_metrics = [
            column for column in metric_candidates if column in display_df.columns
        ]

        for col_name in available_metrics:
            if pd.api.types.is_numeric_dtype(display_df[col_name]) and display_df[col_name].max() <= 1.0:
                display_df[col_name] = display_df[col_name] * 100.0

        with st.container(border=True):
            st.subheader("Held-out test benchmark table", icon=":material/table_chart:")
            st.dataframe(
                display_df,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="#%d", width="small"),
                    "Model": st.column_config.TextColumn("Architecture", pinned=True),
                    "Accuracy": st.column_config.ProgressColumn(
                        "Accuracy",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Precision": st.column_config.NumberColumn("Macro precision", format="%.2f%%"),
                    "Recall": st.column_config.NumberColumn("Macro recall", format="%.2f%%"),
                    "F1 Score": st.column_config.ProgressColumn(
                        "Macro F1 score",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Model Size (MB)": st.column_config.NumberColumn(
                        "Model size",
                        format="%.2f MB",
                    ),
                },
                alt="Benchmark comparison table of accuracy, macro precision, macro recall, macro F1 score, and model size in megabytes",
            )

        if "Model" in display_df.columns and available_metrics:
            with st.container(border=True):
                st.subheader("Evaluation metrics by architecture (%)", icon=":material/bar_chart:")
                long_metrics = display_df.melt(
                    id_vars="Model",
                    value_vars=available_metrics,
                    var_name="Metric",
                    value_name="Score",
                )
                metric_chart = (
                    alt.Chart(long_metrics)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X(
                            "Score:Q",
                            scale=alt.Scale(domain=[0, 100], clamp=True),
                            axis=alt.Axis(title="Score (%)", values=[0, 20, 40, 60, 80, 100]),
                        ),
                        y=alt.Y("Model:N", sort="-x", axis=alt.Axis(title=None)),
                        color=alt.Color("Metric:N", title="Metric"),
                        yOffset="Metric:N",
                        tooltip=[
                            alt.Tooltip("Model:N", title="Architecture"),
                            alt.Tooltip("Metric:N", title="Metric"),
                            alt.Tooltip("Score:Q", title="Score (%)", format=".2f"),
                        ],
                    )
                    .properties(height=280)
                )

                st.altair_chart(
                    metric_chart,
                    alt="Grouped bar chart comparing accuracy, precision, recall, and F1 score across VGG19, VGG16, MobileNetV2, and Custom CNN",
                )

        if "Accuracy" in display_df.columns and "Model" in display_df.columns:
            accuracy_values = pd.to_numeric(display_df["Accuracy"], errors="coerce")
            if accuracy_values.notna().any():
                best_row = display_df.loc[accuracy_values.idxmax()]
                best_accuracy = float(best_row["Accuracy"])
                st.success(
                    f"**{best_row['Model']}** achieved the highest test accuracy at **{best_accuracy:.2f}%**, "
                    "demonstrating the advantage of deep ImageNet transfer learning over training from scratch.",
                    title="Benchmark takeaway",
                    icon=":material/workspace_premium:",
                )

    st.caption(
        "All four models were evaluated on the identical held-out test split and class-index mapping."
    )


# ============================================================
# 7. TRAINING ANALYTICS PAGE
# ============================================================

elif page == "Training Analytics":
    st.badge("Model diagnostics", icon=":material/monitoring:", color="green")
    st.title("Training and evaluation analytics")
    st.markdown(
        "Inspect cross-model comparison plots, per-epoch learning curves, and class-level confusion matrices."
    )

    tab_overview, tab_curves, tab_confusion = st.tabs(
        [
            ":material/compare_arrows: Overall comparisons",
            ":material/timeline: Training curves",
            ":material/grid_on: Confusion matrices",
        ],
        on_change="rerun",
    )

    model_prefix = {
        "VGG19": "vgg19",
        "VGG16": "vgg16",
        "MobileNetV2": "mobilenetv2",
        "Custom CNN": "cnn",
    }

    if tab_overview.open:
        with tab_overview:
            show_output_image(
                "final_metrics_comparison.png",
                "Final metrics comparison",
                "Bar chart comparing accuracy, precision, recall, and F1 score across all four models",
            )
            with st.grid(2, min_column_width=300, gap="small"):
                show_output_image(
                    "final_accuracy_comparison.png",
                    "Accuracy comparison",
                    "Comparison chart of test accuracy across Custom CNN, MobileNetV2, VGG16, and VGG19",
                )
                show_output_image(
                    "final_performance_comparison.png",
                    "Final performance summary",
                    "Multi-metric performance summary chart for the four evaluated architectures",
                )
                show_output_image(
                    "model_size_comparison.png",
                    "Model size comparison (MB)",
                    "Comparison of saved model file sizes in megabytes across architectures",
                )
                show_output_image(
                    "training_time_comparison.png",
                    "Training time comparison (minutes)",
                    "Comparison of total training duration in minutes across architectures",
                )

    if tab_curves.open:
        with tab_curves:
            chosen_model = (
                st.segmented_control(
                    "Select architecture",
                    list(model_prefix.keys()),
                    default="VGG19",
                    key="training_curve_model",
                )
                or "VGG19"
            )
            prefix = model_prefix[chosen_model]
            history_df = load_training_history(prefix)

            if history_df is not None and {"epoch", "accuracy", "val_accuracy", "loss", "val_loss"}.issubset(history_df.columns):
                plot_df = history_df.copy()
                plot_df["Epoch"] = plot_df["epoch"] + 1
                acc_long = plot_df.melt(
                    id_vars="Epoch",
                    value_vars=["accuracy", "val_accuracy"],
                    var_name="Split",
                    value_name="Accuracy",
                )
                acc_long["Split"] = acc_long["Split"].map(
                    {"accuracy": "Training", "val_accuracy": "Validation"}
                )
                acc_long["Accuracy (%)"] = acc_long["Accuracy"] * 100.0

                loss_long = plot_df.melt(
                    id_vars="Epoch",
                    value_vars=["loss", "val_loss"],
                    var_name="Split",
                    value_name="Loss",
                )
                loss_long["Split"] = loss_long["Split"].map(
                    {"loss": "Training", "val_loss": "Validation"}
                )

                with st.grid(2, min_column_width=300, gap="small"):
                    with st.container(border=True, height="stretch"):
                        st.markdown(f"**{chosen_model} interactive accuracy curve**")
                        acc_chart = (
                            alt.Chart(acc_long)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("Epoch:Q", title="Epoch"),
                                y=alt.Y("Accuracy (%):Q", title="Accuracy (%)"),
                                color=alt.Color("Split:N", title="Split"),
                                tooltip=[
                                    alt.Tooltip("Epoch:Q", title="Epoch"),
                                    alt.Tooltip("Split:N", title="Split"),
                                    alt.Tooltip("Accuracy (%):Q", title="Accuracy (%)", format=".2f"),
                                ],
                            )
                            .properties(height=260)
                        )
                        st.altair_chart(
                            acc_chart,
                            alt=f"Line chart of training and validation accuracy across epochs for {chosen_model}",
                        )

                    with st.container(border=True, height="stretch"):
                        st.markdown(f"**{chosen_model} interactive loss curve**")
                        loss_chart = (
                            alt.Chart(loss_long)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("Epoch:Q", title="Epoch"),
                                y=alt.Y("Loss:Q", title="Cross-entropy loss"),
                                color=alt.Color("Split:N", title="Split"),
                                tooltip=[
                                    alt.Tooltip("Epoch:Q", title="Epoch"),
                                    alt.Tooltip("Split:N", title="Split"),
                                    alt.Tooltip("Loss:Q", title="Loss", format=".4f"),
                                ],
                            )
                            .properties(height=260)
                        )
                        st.altair_chart(
                            loss_chart,
                            alt=f"Line chart of training and validation loss across epochs for {chosen_model}",
                        )

            with st.grid(2, min_column_width=300, gap="small"):
                show_output_image(
                    f"{prefix}_accuracy.png",
                    f"{chosen_model} exported accuracy plot",
                    f"Saved training and validation accuracy plot for {chosen_model}",
                )
                show_output_image(
                    f"{prefix}_loss.png",
                    f"{chosen_model} exported loss plot",
                    f"Saved training and validation loss plot for {chosen_model}",
                )

            if history_df is not None:
                with st.expander("View epoch-by-epoch training history table", icon=":material/table_chart:"):
                    st.dataframe(
                        history_df,
                        hide_index=True,
                        alt=f"Training history dataframe by epoch for {chosen_model}",
                    )

    if tab_confusion.open:
        with tab_confusion:
            chosen_cm = (
                st.segmented_control(
                    "Select architecture",
                    list(model_prefix.keys()),
                    default="VGG19",
                    key="confusion_matrix_model",
                )
                or "VGG19"
            )
            cm_prefix = model_prefix[chosen_cm]

            cm_col, report_col = st.columns([1.15, 0.85], gap="medium")
            with cm_col:
                show_output_image(
                    f"{cm_prefix}_confusion_matrix.png",
                    f"{chosen_cm} confusion matrix",
                    f"Confusion matrix heatmap of true versus predicted waste classes for {chosen_cm}",
                )

            with report_col:
                report_text = load_classification_report(cm_prefix)
                with st.container(border=True, height="stretch"):
                    st.markdown(f"**{chosen_cm} per-class classification report**")
                    if report_text:
                        st.code(report_text, language="text")
                    else:
                        st.caption("Classification report file not found in `outputs/`.")


# ============================================================
# 8. PROJECT DETAILS PAGE
# ============================================================

elif page == "Project Details":
    st.badge("Project documentation", icon=":material/info:", color="green")
    st.title("About the project")
    st.markdown(
        "WasteAI is a deep learning research prototype designed to classify waste images "
        "into six material categories and benchmark four convolutional neural network architectures."
    )

    with st.grid(2, min_column_width=300, gap="small"):
        with st.container(border=True, height="stretch"):
            st.badge("Dataset", icon=":material/dataset:", color="green")
            st.subheader("Garbage Classification")
            st.markdown(
                "Benchmark dataset containing six household waste categories: "
                "**cardboard**, **glass**, **metal**, **paper**, **plastic**, and **trash**."
            )
            st.link_button(
                "Open Kaggle dataset",
                "https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification",
                icon=":material/open_in_new:",
            )

        with st.container(border=True, height="stretch"):
            st.badge("Software stack", icon=":material/code:", color="blue")
            st.subheader("TensorFlow, Keras, and Streamlit")
            st.markdown(
                "Trained in Google Colab with TensorFlow/Keras and deployed as an interactive "
                "Streamlit analytics application with Altair, Pandas, NumPy, and Pillow."
            )
            with st.container(horizontal=True):
                st.link_button(
                    "GitHub repository",
                    "https://github.com/Atchyuteswar/WasteAI",
                    icon=":material/code:",
                )
                st.link_button(
                    "Hugging Face model weights",
                    "https://huggingface.co/Atchyuteswar/wasteai-models",
                    icon=":material/cloud_download:",
                )

    with st.container(border=True):
        st.subheader("Architectures evaluated", icon=":material/neurology:")
        architecture_df = pd.DataFrame(
            [
                {
                    "Architecture": name,
                    "Role": config["role"],
                    "Test accuracy": config["test_accuracy"],
                    "Model size (MB)": config["size_mb"],
                    "Preprocessing": config["preprocessing"],
                    "Description": config["description"],
                }
                for name, config in MODEL_CONFIG.items()
            ]
        )
        st.dataframe(
            architecture_df,
            hide_index=True,
            column_config={
                "Architecture": st.column_config.TextColumn("Architecture", pinned=True),
                "Role": st.column_config.TextColumn("Role"),
                "Test accuracy": st.column_config.ProgressColumn(
                    "Recorded test accuracy",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Model size (MB)": st.column_config.NumberColumn(
                    "Model size",
                    format="%.2f MB",
                ),
                "Preprocessing": st.column_config.TextColumn("Input normalization"),
                "Description": st.column_config.TextColumn("Architecture summary", width="large"),
            },
            alt="Summary table of the four evaluated CNN architectures, their test accuracy, model size, preprocessing mode, and description",
        )

    st.warning(
        "This model predicts the six dataset categories only. It does not identify exact "
        "plastic resin codes, food contamination levels, or local municipal recycling eligibility. "
        "Always verify disposal guidance against local collection rules.",
        title="Scope & limitations",
        icon=":material/warning:",
    )

    diag_expander = st.expander(
        "Environment & model availability diagnostics",
        icon=":material/terminal:",
        on_change="rerun",
    )
    if diag_expander.open:
        with diag_expander:
            diag_data: dict[str, str] = {
                ":material/memory: TensorFlow version": f"`{tf.__version__}`",
                ":material/web: Streamlit version": f"`{st.__version__}`",
                ":material/folder: Local models directory": (
                    ":green-badge[Present]" if MODEL_DIR.exists() else ":orange-badge[Missing]"
                ),
                ":material/folder_open: Local outputs directory": (
                    ":green-badge[Present]" if OUTPUT_DIR.exists() else ":orange-badge[Missing]"
                ),
                ":material/cloud: Hugging Face repository": (
                    f"`{HF_MODEL_REPO}`" if HF_MODEL_REPO else ":gray-badge[Not configured]"
                ),
            }
            for model_name, config in MODEL_CONFIG.items():
                model_file = MODEL_DIR / config["filename"]
                if model_file.exists():
                    status_badge = ":green-badge[Local file ready]"
                elif HF_MODEL_REPO:
                    status_badge = ":blue-badge[Hugging Face Hub]"
                else:
                    status_badge = ":orange-badge[Not found locally]"
                diag_data[f":material/model_training: {model_name} (`{config['filename']}`)"] = status_badge

            st.table(
                diag_data,
                border="horizontal",
                alt="Environment and model file availability diagnostics",
            )
