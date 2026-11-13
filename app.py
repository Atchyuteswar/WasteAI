"""
WasteAI — Deep Learning Waste Classification
A fresh Streamlit application (not a patch to the previous app).

Run:
    streamlit run app.py

Expected project layout:
    Waste_Classification/
      app.py
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

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WasteAI | Waste Intelligence",
    page_icon="♻️",
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

MODEL_CONFIG = {
    "VGG19": {
        "filename": "vgg19_model.keras",
        "preprocessing": "vgg19",
        "test_accuracy": 91.29,
        "description": "Deep 19-layer VGG architecture with ImageNet transfer learning.",
    },
    "VGG16": {
        "filename": "vgg16_model.keras",
        "preprocessing": "vgg16",
        "test_accuracy": 88.65,
        "description": "16-layer VGG architecture with ImageNet transfer learning.",
    },
    "MobileNetV2": {
        "filename": "mobilenetv2_model.keras",
        "preprocessing": "mobilenet_v2",
        "test_accuracy": 87.07,
        "description": "Efficient depthwise-separable CNN architecture.",
    },
    "Custom CNN": {
        "filename": "cnn_model.keras",
        "preprocessing": "rescale",
        "test_accuracy": 73.09,
        "description": "Custom four-block convolutional neural network.",
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

DISPOSAL_GUIDE = {
    "cardboard": (
        "Usually recyclable",
        "Keep cardboard clean and dry. Flatten boxes and remove excess tape where practical. "
        "Check your local collection rules."
    ),
    "glass": (
        "Often recyclable",
        "Empty and rinse containers. Accepted glass types vary by local recycling service. "
        "Broken glass may need separate handling."
    ),
    "metal": (
        "Often recyclable",
        "Empty and rinse cans or containers. Check local rules for accepted metal packaging."
    ),
    "paper": (
        "Usually recyclable",
        "Keep paper clean and dry. Greasy or food-soiled paper may not be accepted with ordinary paper."
    ),
    "plastic": (
        "Depends on plastic type and local facilities",
        "Recycling depends on resin type, item shape, contamination and local facilities. "
        "Check the packaging and local collection guidance."
    ),
    "trash": (
        "General-waste dataset category",
        "The model's 'trash' label is a dataset category, not a definitive material analysis. "
        "Check whether the item can be reused, separated, recycled or composted."
    ),
}


# ============================================================
# 2. VISUAL DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --green-950: #10291d;
        --green-900: #153b29;
        --green-700: #1b7048;
        --green-600: #258456;
        --green-100: #eaf3ed;
        --ink: #17251d;
        --muted: #66766c;
        --line: #e1e9e3;
        --canvas: #f5f8f5;
        --white: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(245,248,245,.94);
    }

    [data-testid="stSidebar"] {
        background: var(--green-950);
        border-right: 1px solid #244b35;
    }

    [data-testid="stSidebar"] * {
        color: #eff7f0;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 9px;
        padding: 7px 9px;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
        letter-spacing: -0.35px;
    }

    h1 { font-weight: 800; letter-spacing: -1px; }
    h2, h3 { font-weight: 700; }

    .hero {
        background: linear-gradient(115deg, #103b28 0%, #1d754b 100%);
        border-radius: 20px;
        padding: 34px 36px;
        color: white;
        margin: 4px 0 24px 0;
        box-shadow: 0 12px 32px rgba(16, 59, 40, .12);
    }

    .hero-kicker {
        color: #bce0c8;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.12;
        margin: 10px 0;
    }

    .hero p {
        color: #e1f0e5;
        font-size: 1rem;
        max-width: 760px;
        margin: 0;
    }

    .section-kicker {
        color: #62816e;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .surface {
        background: var(--white);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 4px 16px rgba(23, 55, 35, .035);
        height: 100%;
    }

    .stat-label {
        color: var(--muted);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .06em;
        text-transform: uppercase;
    }

    .stat-value {
        color: var(--ink);
        font: 800 1.85rem 'Manrope', sans-serif;
        margin: 5px 0 1px;
    }

    .stat-note {
        color: var(--muted);
        font-size: .78rem;
    }

    .result-panel {
        background: #edf5ef;
        border: 1px solid #d7e8dc;
        border-radius: 16px;
        padding: 22px;
    }

    .result-class {
        color: #164d32;
        font: 800 2.2rem 'Manrope', sans-serif;
        text-transform: capitalize;
    }

    .tag {
        display: inline-block;
        background: #e7f2ea;
        color: #185536;
        border-radius: 999px;
        padding: 5px 11px;
        font-size: .75rem;
        font-weight: 700;
    }

    .small-muted {
        color: var(--muted);
        font-size: .82rem;
    }

    div.stButton > button[kind="primary"] {
        background: var(--green-700);
        border: 1px solid var(--green-700);
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"]:hover {
        background: var(--green-900);
        border-color: var(--green-900);
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #a9c5b1;
        border-radius: 14px;
        padding: 10px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    .footer {
        color: #7b897f;
        font-size: .78rem;
        text-align: center;
        padding: 18px 0 4px;
    }

    hr { border-color: var(--line); }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATA AND MODEL HELPERS
# ============================================================

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
    """
    Preprocess one RGB image.

    The preprocessing mode must match the exact preprocessing used in Colab.
    Review the comment above MODEL_CONFIG if your training pipeline used a
    different rescaling/preprocess_input sequence.
    """
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


def predict_image(image: Image.Image, model_name: str):
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


def stat_card(label: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="surface">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_output_image(filename: str, title: str):
    path = OUTPUT_DIR / filename
    st.markdown(f"#### {title}")

    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.info(f"Visualization not available: outputs/{filename}")


def load_comparison_data() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "final_model_comparison.csv"
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.error(f"Could not read model comparison CSV: {error}")
        return None


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 2px 16px">
            <div style="font:800 1.55rem Manrope;color:#ffffff">♻ WasteAI</div>
            <div style="font-size:.68rem;letter-spacing:.13em;color:#b4cdbb;font-weight:700">
                WASTE INTELLIGENCE PLATFORM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Waste Classifier",
            "Model Performance",
            "Training Analytics",
            "Project Details",
        ],
        label_visibility="visible",
    )

    st.divider()

    st.markdown("**PROJECT SNAPSHOT**")
    st.markdown(f"♻️ `{len(CLASSES):02d}` material classes")
    st.markdown("🧠 `04` trained architectures")
    st.markdown("🏆 `91.29%` best recorded accuracy")
    st.caption("Academic research prototype")


# ============================================================
# 5. OVERVIEW PAGE
# ============================================================

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">AI-POWERED MATERIAL RECOGNITION</div>
            <h1>Smarter waste recognition.<br>Better-informed decisions.</h1>
            <p>
                An image-based waste classification research prototype that evaluates
                four deep learning architectures across six material categories.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(4)
    overview_stats = [
        ("MATERIAL CLASSES", f"{len(CLASSES):02d}", "Six waste image categories"),
        ("MODELS EVALUATED", "04", "CNN, MobileNetV2, VGG16, VGG19"),
        ("BEST TEST ACCURACY", "91.29%", "Recorded VGG19 result"),
        ("TOP-RANKED MODEL", "VGG19", "Highest recorded accuracy"),
    ]

    for column, (label, value, note) in zip(columns, overview_stats):
        with column:
            stat_card(label, value, note)

    st.markdown("### Explore WasteAI")
    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="surface">
                <div class="section-kicker">01 · IMAGE CLASSIFICATION</div>
                <h3>Identify a waste category</h3>
                <p class="small-muted">
                    Upload an image, select one of the trained models and inspect
                    the predicted class and score distribution.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="surface">
                <div class="section-kicker">02 · EXPERIMENT ANALYSIS</div>
                <h3>Compare model performance</h3>
                <p class="small-muted">
                    Explore test metrics, training curves, confusion matrices
                    and model comparison visualizations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.info(
        "The reported model metrics come from the saved test experiments. "
        "They do not guarantee the same performance on every new image."
    )


# ============================================================
# 6. WASTE CLASSIFIER PAGE
# ============================================================

elif page == "Waste Classifier":
    st.markdown('<div class="section-kicker">IMAGE INFERENCE</div>', unsafe_allow_html=True)
    st.title("Classify a waste item")
    st.write("Upload a clear image of one item to generate a model prediction.")

    upload_col, preview_col = st.columns([0.95, 1.05], gap="large")

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload waste image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Use a clear, well-lit image with the main object visible.",
        )

        selected_model = st.selectbox(
            "Select classification model",
            list(MODEL_CONFIG.keys()),
            index=0,
        )

        st.caption(MODEL_CONFIG[selected_model]["description"])

        classify_button = st.button(
            "♻  Analyze waste image",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

    uploaded_image = None

    with preview_col:
        if uploaded_file is not None:
            try:
                uploaded_image = Image.open(uploaded_file)
                uploaded_image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
                st.image(
                    uploaded_image,
                    caption="Image selected for classification",
                    use_container_width=True,
                )
            except Exception:
                st.error("The uploaded image could not be opened. Please try another image.")
        else:
            st.markdown(
                """
                <div class="surface" style="height:270px;display:flex;align-items:center;justify-content:center;text-align:center">
                    <div>
                        <div style="font-size:2.5rem">▧</div>
                        <h3>Image preview</h3>
                        <p class="small-muted">Your selected image will appear here.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if classify_button and uploaded_image is not None:
        try:
            started = time.perf_counter()

            with st.spinner(f"Analyzing image with {selected_model}..."):
                scores, ranking = predict_image(uploaded_image, selected_model)

            elapsed = time.perf_counter() - started
            best_index = int(ranking[0])
            predicted_class = CLASSES[best_index]
            top_score = float(scores[best_index])

            guide_title, guide_text = DISPOSAL_GUIDE.get(
                predicted_class,
                ("Check local guidance", "Follow local waste collection guidance."),
            )

            st.divider()
            st.markdown("### Classification result")

            result_left, result_right = st.columns([1.2, 0.8])

            with result_left:
                st.markdown(
                    f"""
                    <div class="result-panel">
                        <div class="tag">MODEL PREDICTION</div>
                        <div class="result-class">{predicted_class}</div>
                        <div style="color:#426650;font-weight:700;margin-top:5px">
                            Top model score: {top_score * 100:.2f}%
                        </div>
                        <p class="small-muted" style="margin-top:12px">
                            Predicted using {selected_model}.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with result_right:
                stat_card("INFERENCE TIME", f"{elapsed:.2f}s", selected_model)
                st.markdown(
                    f"""
                    <div class="surface" style="margin-top:12px">
                        <div class="stat-label">DISPOSAL GUIDANCE</div>
                        <div style="font-weight:800;color:#185536;margin-top:7px">{guide_title}</div>
                        <div class="stat-note" style="margin-top:6px">{guide_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if top_score < 0.60:
                st.warning(
                    "The top model score is low. Treat this prediction as uncertain "
                    "and verify the item manually."
                )
            elif top_score < 0.80:
                st.info(
                    "The model has a moderate top score. Verify the material before "
                    "making a disposal decision."
                )

            st.markdown("### Score distribution across all classes")

            score_table = pd.DataFrame(
                {
                    "Material": [CLASSES[index].title() for index in ranking],
                    "Score (%)": [float(scores[index]) * 100 for index in ranking],
                }
            )

            st.bar_chart(score_table.set_index("Material"), horizontal=True)

            with st.expander("View detailed prediction scores"):
                st.dataframe(score_table, use_container_width=True, hide_index=True)

            st.caption(
                "The displayed score is the model's output score. It is not necessarily "
                "a calibrated probability of correctness."
            )

        except Exception as error:
            st.error("The selected model could not complete classification.")
            st.exception(error)
            st.markdown(
                """
                **Troubleshooting checks**
                - Confirm the selected `.keras` file exists in `models/`.
                - Confirm local TensorFlow/Keras versions are compatible with the saved model.
                - Confirm the class mapping has six entries.
                - Confirm inference preprocessing matches the exact Colab training pipeline.
                """
            )


# ============================================================
# 7. MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":
    st.markdown('<div class="section-kicker">EXPERIMENT RESULTS</div>', unsafe_allow_html=True)
    st.title("Model performance comparison")
    st.write("Review the four architectures using the saved held-out test results.")

    comparison_df = load_comparison_data()

    if comparison_df is None:
        st.warning(
            "The file `outputs/final_model_comparison.csv` was not found. "
            "Copy the Stage 7 comparison CSV from Google Drive into the local outputs folder."
        )
    else:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        metric_candidates = ["Accuracy", "Precision", "Recall", "F1 Score"]
        available_metrics = [
            column for column in metric_candidates
            if column in comparison_df.columns
        ]

        if "Model" in comparison_df.columns and available_metrics:
            chart_df = comparison_df.set_index("Model")[available_metrics].copy()

            # Stage 7 results were saved as proportions (0.9129), but support
            # percentage-formatted CSVs too.
            if chart_df.max().max() <= 1.0:
                chart_df = chart_df * 100

            st.markdown("### Evaluation metrics (%)")
            st.bar_chart(chart_df)

        if "Accuracy" in comparison_df.columns and "Model" in comparison_df.columns:
            accuracy_values = pd.to_numeric(
                comparison_df["Accuracy"], errors="coerce"
            )
            if accuracy_values.notna().any():
                best_row = comparison_df.loc[accuracy_values.idxmax()]
                best_accuracy = float(best_row["Accuracy"])
                if best_accuracy <= 1:
                    best_accuracy *= 100

                st.success(
                    f"Highest recorded accuracy: {best_row['Model']} "
                    f"({best_accuracy:.2f}%)."
                )

    st.caption(
        "A fair comparison requires every model to use the same held-out test images "
        "and the same class-index mapping."
    )


# ============================================================
# 8. TRAINING ANALYTICS PAGE
# ============================================================

elif page == "Training Analytics":
    st.markdown('<div class="section-kicker">MODEL DIAGNOSTICS</div>', unsafe_allow_html=True)
    st.title("Training and evaluation analytics")

    tab_overview, tab_curves, tab_confusion = st.tabs(
        ["Overall comparisons", "Training curves", "Confusion matrices"]
    )

    with tab_overview:
        show_output_image(
            "final_metrics_comparison.png",
            "Final metrics comparison",
        )
        show_output_image(
            "final_accuracy_comparison.png",
            "Accuracy comparison",
        )
        show_output_image(
            "final_performance_comparison.png",
            "Final performance comparison",
        )
        show_output_image(
            "model_size_comparison.png",
            "Model size comparison",
        )
        show_output_image(
            "training_time_comparison.png",
            "Training time comparison",
        )

    model_prefix = {
        "Custom CNN": "cnn",
        "MobileNetV2": "mobilenetv2",
        "VGG16": "vgg16",
        "VGG19": "vgg19",
    }

    with tab_curves:
        chosen_model = st.selectbox(
            "Choose model",
            list(model_prefix.keys()),
            key="training_curve_model",
        )
        prefix = model_prefix[chosen_model]

        curve_left, curve_right = st.columns(2)
        with curve_left:
            show_output_image(
                f"{prefix}_accuracy.png",
                f"{chosen_model} accuracy curve",
            )
        with curve_right:
            show_output_image(
                f"{prefix}_loss.png",
                f"{chosen_model} loss curve",
            )

        history_path = OUTPUT_DIR / f"{prefix}_training_history.csv"
        if history_path.exists():
            with st.expander("View training history data"):
                try:
                    history_df = pd.read_csv(history_path)
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
                except Exception as error:
                    st.error(f"Could not read training history: {error}")

    with tab_confusion:
        chosen_cm = st.selectbox(
            "Choose confusion matrix",
            list(model_prefix.keys()),
            key="confusion_matrix_model",
        )
        cm_prefix = model_prefix[chosen_cm]
        show_output_image(
            f"{cm_prefix}_confusion_matrix.png",
            f"{chosen_cm} confusion matrix",
        )

        report_path = OUTPUT_DIR / f"{cm_prefix}_classification_report.txt"
        if report_path.exists():
            with st.expander("View classification report"):
                st.code(report_path.read_text(encoding="utf-8"))


# ============================================================
# 9. PROJECT DETAILS PAGE
# ============================================================

elif page == "Project Details":
    st.markdown('<div class="section-kicker">PROJECT DOCUMENTATION</div>', unsafe_allow_html=True)
    st.title("About the project")

    st.write(
        "WasteAI is a deep learning-based academic prototype designed to classify "
        "waste images into six categories and compare the performance of four CNN architectures."
    )

    details_left, details_right = st.columns(2)

    with details_left:
        st.markdown(
            """
            <div class="surface">
                <div class="section-kicker">DATASET</div>
                <h3>Garbage Classification</h3>
                <p class="small-muted">
                    Six image categories: cardboard, glass, metal, paper, plastic and trash.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button(
            "Open Kaggle dataset",
            "https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification",
        )

    with details_right:
        st.markdown(
            """
            <div class="surface">
                <div class="section-kicker">FRAMEWORK</div>
                <h3>TensorFlow and Keras</h3>
                <p class="small-muted">
                    Streamlit application with NumPy, Pandas, Pillow, Matplotlib,
                    Seaborn and Scikit-learn.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Architectures evaluated")
    architecture_df = pd.DataFrame(
        [
            {"Architecture": name, "Recorded test accuracy": f"{config['test_accuracy']:.2f}%", "Description": config["description"]}
            for name, config in MODEL_CONFIG.items()
        ]
    )
    st.dataframe(architecture_df, use_container_width=True, hide_index=True)

    st.markdown("### Classification categories")
    st.write(" · ".join(name.title() for name in CLASSES))

    st.warning(
        "This model predicts the six dataset categories. It does not identify exact "
        "plastic resin composition, contamination level or local recycling eligibility. "
        "Disposal guidance is general and must be checked against local municipal rules."
    )

    with st.expander("Environment diagnostics"):
        st.write(f"Python runtime: {tf.sysconfig.get_build_info().get('python_version', 'See local Python installation')}")
        st.write(f"TensorFlow: {tf.__version__}")
        st.write(f"Model directory exists: {MODEL_DIR.exists()}")
        st.write(f"Output directory exists: {OUTPUT_DIR.exists()}")

        st.write(f"Hugging Face model repo configured: {'Yes' if HF_MODEL_REPO else 'No'}")
        for model_name, config in MODEL_CONFIG.items():
            model_file = MODEL_DIR / config["filename"]
            source = "local" if model_file.exists() else ("Hugging Face" if HF_MODEL_REPO else "not configured")
            st.write(f"{model_name}: {source}")


# ============================================================
# 10. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        WasteAI · Deep Learning-Based Waste Classification · Academic Research Prototype
    </div>
    """,
    unsafe_allow_html=True,
)
