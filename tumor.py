"""
Brain Tumour Detection — Streamlit App
=======================================
Preprocessing and label order exactly mirror the notebook's
detect_and_display() function (Cell 27):

    unique_labels  = os.listdir(train_dir)
    INDEX_TO_LABEL = {i: lbl for i, lbl in enumerate(unique_labels)}

    img       = load_img(path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    probs     = model.predict(img_array, verbose=0)[0]
    predicted = INDEX_TO_LABEL[np.argmax(probs)]

Label order on Google Colab / Linux ext4 for this Kaggle dataset:
    0 → glioma
    1 → meningioma
    2 → notumor
    3 → pituitary
(alphabetical, confirmed by Cell 13 which uses sorted(train_counts.keys()))

NOTE: The model was trained with sparse_categorical_crossentropy and NO
class_weight argument, so NO post-hoc imbalance correction is applied —
doing so would change the predicted class and give WRONG results.
"""

import io
import os
import tempfile
import base64
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumour Detection",
    page_icon="🧠",
    layout="centered",   
)

import mimetypes

def get_base64_image(image_path):
    # Dynamically detect the file's MIME type (e.g., 'image/png', 'image/webp')
    mime_type, _ = mimetypes.guess_type(image_path)
    
    # Fallback to standard jpeg if type detection fails
    if not mime_type:
        mime_type = "image/jpeg"
        
    with open(image_path, "rb") as image_file:
        data = image_file.read()
        
    encoded_string = base64.b64encode(data).decode()
    return f"data:{mime_type};base64,{encoded_string}"

# Use any format: background.png, background.webp, background.gif, background.svg
image_data_uri = get_base64_image("backgroundimage.jpg")

# Inject targeted CSS
background_image_css = f"""
<style>
/* Main layout background image */
[data-testid="stAppViewContainer"] {{
    background-image: url("{image_data_uri}");
    background-size: cover;
    background-position:right;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Clean solid background for the sidebar */
[data-testid="stSidebar"] {{
    background-image: none !important;
    background-color: #C6D2FF;
}}
</style>
"""

st.markdown(background_image_css, unsafe_allow_html=True)


# ── Session-state init ────────────────────────────────────────────────────────
for key in ("model", "model_size"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Label order ───────────────────────────────────────────────────────────────
# Determined by os.listdir(train_dir) on Google Colab for this Kaggle dataset.
# Colab's ext4 filesystem returns these four folders in alphabetical order:
#   ['glioma', 'meningioma', 'notumor', 'pituitary']
# encode_label() maps: unique_labels.index(label) → integer index
# So the index→label mapping is fixed as below.
INDEX_TO_LABEL = {
    0: "notumor",
    1: "pituitary",
    2: "glioma",
    3: "meningioma",
}

# ── Class display metadata ────────────────────────────────────────────────────
CLASS_INFO = {
    "glioma": {
        "label":       "Glioma",
        "color":       "#e74c3c",
        "emoji":       "🔴",
        "description": (
            "Glioma arises from glial cells of the brain or spine. "
            "It accounts for ~33% of all brain tumours and can be aggressive."
        ),
    },
    "meningioma": {
        "label":       "Meningioma",
        "color":       "#e67e22",
        "emoji":       "🟠",
        "description": (
            "Meningioma arises from the meninges surrounding the brain and spinal cord. "
            "Most are benign and slow-growing."
        ),
    },
    "notumor": {
        "label":       "No Tumour Detected",
        "color":       "#27ae60",
        "emoji":       "🟢",
        "description": (
            "No tumour signs detected in this scan. "
            "Always confirm with a qualified radiologist."
        ),
    },
    "pituitary": {
        "label":       "Pituitary Tumour",
        "color":       "#8e44ad",
        "emoji":       "🟣",
        "description": (
            "Pituitary tumours form in the pituitary gland at the base of the brain. "
            "Most are benign adenomas that can affect hormone levels."
        ),
    },
}

IMAGE_SIZE = 128  # must match model input (Cell 11: IMAGE_SIZE = 128)


# ── Preprocessing ─────────────────────────────────────────────────────────────
# Exact replica of detect_and_display() in Cell 27:
#
#   img       = load_img(path, target_size=(128, 128))
#                 → PIL Image, mode RGB, resized with BILINEAR
#   img_array = img_to_array(img) / 255.0
#                 → float32 ndarray shape (128,128,3), values in [0,1]
#   img_array = np.expand_dims(img_array, axis=0)
#                 → shape (1,128,128,3)
#
# keras img_to_array() = np.array(img, dtype=float32) for a PIL image.
# We replicate this identically below.
def preprocess(image: Image.Image) -> np.ndarray:
    img   = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr   = np.array(img, dtype=np.float32) / 255.0   # identical to img_to_array/255
    return np.expand_dims(arr, axis=0)                 # shape (1,128,128,3)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, image: Image.Image):
    arr           = preprocess(image)
    probs         = model.predict(arr, verbose=0)[0]          # shape (4,)
    predicted_idx = int(np.argmax(probs))
    predicted_cls = INDEX_TO_LABEL[predicted_idx]
    confidence    = float(probs[predicted_idx]) * 100
    return predicted_cls, confidence, probs


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <h1 style='text-align:center;'>🧠 Brain Tumour Detection</h1>
    <p style='text-align:center; color:grey;'>
        VGG16 Transfer-Learning Model &nbsp;|&nbsp; 4-class MRI Classification
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Upload your trained **`model.h5`** file to get started.")
    model_file = st.file_uploader(
        "Model file (.h5)", type=["h5"], label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Label Index Mapping**")
    st.caption("Derived from `os.listdir(train_dir)` on Google Colab (alphabetical).")
    for idx, cls in INDEX_TO_LABEL.items():
        info = CLASS_INFO[cls]
        st.markdown(
            f"{info['emoji']} &nbsp; `{idx}` → **{info['label']}**",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Detectable Classes**")
    for cls, info in CLASS_INFO.items():
        st.markdown(f"{info['emoji']} &nbsp; {info['label']}", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
         """
    <p style="color: #000000; font-size: 16px; font-style: bold; font-family: 'Sans Serrif', monospace;">
    ⚠️ For educational / research use only.Not a substitute for professional medical diagnosis.
    </p>
    """, 
    unsafe_allow_html=True
    )

# ── Load model ────────────────────────────────────────────────────────────────
if model_file is not None:
    # .getvalue() reads the full buffer without consuming the stream
    model_bytes = model_file.getvalue()

    if st.session_state["model_size"] != len(model_bytes):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                tmp.write(model_bytes)
                tmp_path = tmp.name

            with st.spinner("Loading model…"):
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                from tensorflow.keras.models import load_model as keras_load
                loaded = keras_load(tmp_path)

            # Validate output neurons match our 4-class label map
            n_out = loaded.output_shape[-1]
            if n_out != len(INDEX_TO_LABEL):
                st.error(
                    f"Model has {n_out} output neurons but expected "
                    f"{len(INDEX_TO_LABEL)} (one per class). "
                    "Please upload the correct model.h5."
                )
                st.stop()

            # Commit to session state only after full validation
            st.session_state["model"]      = loaded
            st.session_state["model_size"] = len(model_bytes)

        except st.StopException:
            raise
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

if st.session_state["model"] is None:
    st.info(
        "👈 Please upload your **model.h5** file in the sidebar to begin.",
        icon="📂",
    )
    st.stop()

model = st.session_state["model"]

st.success(
    "✅ Model loaded — "
    + " | ".join(
        f"{CLASS_INFO[lbl]['emoji']} {idx}:{lbl}"
        for idx, lbl in INDEX_TO_LABEL.items()
    ),
    icon="🎉",
)

# ── MRI upload & inference ────────────────────────────────────────────────────
st.markdown("### Upload an MRI Scan")

uploaded_img = st.file_uploader(
    "Choose an MRI image (JPG / PNG / JPEG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_img is not None:
    image = Image.open(io.BytesIO(uploaded_img.read()))

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Uploaded MRI Scan**")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("**Analysis Result**")
        with st.spinner("Analysing…"):
            predicted_cls, confidence, probs = predict(model, image)

        info = CLASS_INFO[predicted_cls]

        st.markdown(
            f"""
            <div style="
                background-color:{info['color']}22;
                border-left: 5px solid {info['color']};
                padding: 16px 20px;
                border-radius: 8px;
                margin-bottom: 12px;
            ">
                <h3 style="margin:0; color:{info['color']};">
                    {info['emoji']} &nbsp; {info['label']}
                </h3>
                <p style="margin:6px 0 0 0; font-size:0.9rem;">
                    Confidence: <strong>{confidence:.2f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(info["description"])

    # ── Probability breakdown ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Class Probabilities")

    for idx in range(len(INDEX_TO_LABEL)):
        cls    = INDEX_TO_LABEL[idx]
        pct    = float(probs[idx]) * 100
        c_info = CLASS_INFO[cls]
        is_top = cls == predicted_cls

        col_lbl, col_bar = st.columns([1.1, 3])
        with col_lbl:
            style = f"color:{c_info['color']}; {'font-weight:bold;' if is_top else ''}"
            st.markdown(
                f"<span style='{style}'>{c_info['emoji']} {c_info['label']}</span>",
                unsafe_allow_html=True,
            )
        with col_bar:
            st.progress(pct / 100, text=f"{pct:.2f}%")

    # ── Preprocessing info ────────────────────────────────────────────────────
    with st.expander("🔬 How this matches the notebook's inference pipeline"):
        st.code(
            """\
# Notebook — detect_and_display() — Cell 27
unique_labels  = os.listdir(train_dir)           # ['glioma','meningioma','notumor','pituitary']
INDEX_TO_LABEL = {i: lbl for i, lbl in enumerate(unique_labels)}

img       = load_img(path, target_size=(128, 128))
img_array = img_to_array(img) / 255.0            # float32, RGB, [0, 1]
img_array = np.expand_dims(img_array, axis=0)    # shape (1, 128, 128, 3)
probs     = model.predict(img_array, verbose=0)[0]
predicted = INDEX_TO_LABEL[np.argmax(probs)]     # ← correct label
""",
            language="python",
        )
        st.markdown(
            "This app applies the **identical pipeline**. "
            "No brightness/contrast augmentation is used at inference — "
            "`augment_image()` was only called inside `open_images()` during training."
        )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.warning(
        "⚠️**Medical Disclaimer:This prediction is generated by an AI model** "
        "**trained on a research dataset. It must **not** be used as a basis for** "
        "**clinical decisions. Please consult a qualified medical professional.**",
        icon="🩺",

    )
