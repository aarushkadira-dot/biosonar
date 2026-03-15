import os
import json
import base64
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import tempfile
import io

# run from repo root: cd ~/BioSonar && streamlit run app/ui.py
MODEL_PATH   = "model/biosonar.pth"
CLASSES_PATH = "model/classes.json"
CM_PATH      = "model/confusion_matrix.png"
F1_PATH      = "model/f1_scores.png"
LC_PATH      = "model/learning_curve.png"

# mel params must match make_spectrograms.py exactly
# if these change the model breaks silently (wrong input distribution)
SR       = 22050
N_MELS   = 128
HOP      = 512
FMAX     = 8000
IMG_SIZE = 224

# below this we warn instead of showing a confident prediction
# tried 0.6 but it was too lenient, 0.75 is more honest
CONF_THRESHOLD = 0.75

SPECIES_INFO = {
    "beaked_whale": {
        "common": "Beaked Whale",
        "sound": "high-freq clicks above 20kHz — hard to capture at standard sample rates",
        "habitat": "deep offshore waters, usually >1000m",
        "conservation": "vulnerable to navy sonar, strandings linked to military exercises",
        "fun_fact": "longest recorded dive of any mammal — over 2 hours",
    },
    "blue_whale": {
        "common": "Blue Whale",
        "sound": "extremely low 10-40Hz pulses, audible hundreds of miles away",
        "habitat": "open ocean, all major oceans",
        "conservation": "endangered, ~10-25k remaining",
        "fun_fact": "loudest animal on earth, calls reach 188 decibels",
    },
    "false_killer_whale": {
        "common": "False Killer Whale",
        "sound": "whistles and clicks similar to orca but distinct pattern",
        "habitat": "tropical and subtropical deep water",
        "conservation": "near threatened, interactions with longline fishing a major issue",
        "fun_fact": "one of the few cetaceans that shares food with other individuals",
    },
    "fin_whale": {
        "common": "Fin Whale",
        "sound": "20Hz pulses in long sequences, often in pairs",
        "habitat": "deep offshore, both polar and tropical",
        "conservation": "vulnerable, still hunted in small numbers",
        "fun_fact": "second largest animal ever, up to 27m",
    },
    "humpback_whale": {
        "common": "Humpback Whale",
        "sound": "complex songs up to 30 mins, males only, evolve across breeding seasons",
        "habitat": "coastal and shelf waters, migrates pole to equator",
        "conservation": "least concern, recovered well post-whaling ban",
        "fun_fact": "songs spread west to east across ocean basins over years",
    },
    "orca": {
        "common": "Orca (Killer Whale)",
        "sound": "clicks, whistles, pulsed calls — dialect varies by pod",
        "habitat": "all oceans, especially productive coastal areas",
        "conservation": "data deficient overall, some populations critically endangered",
        "fun_fact": "each pod has unique calls passed down through generations",
    },
    "pilot_whale": {
        "common": "Pilot Whale",
        "sound": "whistles and clicks, highly social vocalizations",
        "habitat": "deep temperate and tropical waters",
        "conservation": "least concern, but hunted in Faroe Islands",
        "fun_fact": "one of the most common mass stranding species",
    },
    "sperm_whale": {
        "common": "Sperm Whale",
        "sound": "loud clicks (codas) used for echolocation and communication",
        "habitat": "deep ocean worldwide, dives to 3000m for squid",
        "conservation": "vulnerable, population recovering since whaling ban",
        "fun_fact": "largest brain of any animal ever, ~8kg",
    },
}

BG_COLOR = "#0a0f1e"


def logo_b64():
    path = os.path.join(os.getcwd(), "app", "Logo.png")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def logo_img():
    path = os.path.join(os.getcwd(), "app", "Logo.png")
    return Image.open(path) if os.path.exists(path) else None


@st.cache_resource
def load_model():
    # cache so streamlit doesnt reload 85mb weights on every rerun
    if not os.path.exists(MODEL_PATH):
        return None, None
    if not os.path.exists(CLASSES_PATH):
        return None, None

    with open(CLASSES_PATH) as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    n_classes = len(class_to_idx)

    # exact same arch as train.py - weights wont load if this doesnt match
    backbone = models.resnet34(weights=None)
    feats    = nn.Sequential(*list(backbone.children())[:-1])
    drop     = nn.Dropout(0.3)
    fc       = nn.Linear(512, n_classes)

    class BioSonarNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = feats
            self.dropout  = drop
            self.fc       = fc

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            x = self.dropout(x)
            return self.fc(x)

    model = BioSonarNet()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, idx_to_class


def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    if len(y) < SR:
        return None, "audio too short — need at least 1 second"

    # take middle 5s window, same as slice_audio.py
    target = SR * 5
    if len(y) > target:
        start = (len(y) - target) // 2
        y = y[start:start + target]

    mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # render to png then reload as PIL - same pipeline as make_spectrograms.py
    fig, ax = plt.subplots(figsize=(224/100, 224/100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_position([0, 0, 1, 1])
    librosa.display.specshow(mel_db, sr=SR, hop_length=HOP, fmax=FMAX, ax=ax)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)

    img = Image.open(buf).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return img, None


def predict(model, idx_to_class, img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_idx   = probs.argmax().item()
    top_prob  = probs[top_idx].item()
    top_class = idx_to_class[top_idx]
    all_p = {idx_to_class[i]: probs[i].item() for i in range(len(probs))}

    return top_class, top_prob, all_p


def prob_chart(all_p):
    species = [SPECIES_INFO[k]["common"] for k in sorted(all_p.keys())]
    probs   = [all_p[k] for k in sorted(all_p.keys())]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    colors = ["#00ff9d" if p == max(probs) else "#00d4ff" for p in probs]
    ax.barh(species, probs, color=colors, height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", color="#e8f4f8")
    ax.tick_params(colors="#e8f4f8")
    ax.spines[:].set_color("#1a2540")
    ax.axvline(CONF_THRESHOLD, color="#ff6b6b", linewidth=1,
               linestyle="--", label=f"confidence threshold ({CONF_THRESHOLD})")
    ax.legend(fontsize=8, facecolor=BG_COLOR, edgecolor="#1a2540", labelcolor="#e8f4f8")

    plt.tight_layout()
    return fig


def page_home():
    b64 = logo_b64()
    logo_html = (
        f'<img src="data:image/png;base64,{b64}" width="90" style="vertical-align: middle; margin-right: 14px;">'
        if b64 else ""
    )
    st.markdown(
        f"""
        <div style='text-align: center; padding: 2rem 0 1rem 0;'>
            <h1 style='font-size: 3rem; color: #00d4ff; letter-spacing: 2px;'>
                {logo_html} BioSonar
            </h1>
            <p style='font-size: 1.2rem; color: #e8f4f8; opacity: 0.8;'>
                whale species classification from passive acoustic data
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("The Problem")
        st.write("""
        Marine biologists and organizations like NOAA deploy underwater hydrophones
        that record thousands of hours of ocean audio. Most of it never gets reviewed 
        — there's simply too much for humans to listen through manually.

        Whale vocalizations are buried in hours of noise. Missing them means missing
        data on population health, migration patterns, and responses to human activity
        like shipping and sonar.
        """)

        st.subheader("What BioSonar Does")
        st.write("""
        BioSonar converts raw audio into mel-spectrograms and runs them through a
        fine-tuned ResNet-34 CNN to identify which of 8 whale species is vocalizing.

        Crucially, it flags low-confidence detections instead of guessing —
        so researchers know when to trust the output and when to listen themselves.
        """)

    with col2:
        st.subheader("Model Stats")
        st.metric("Test Accuracy", "99.6%")
        st.metric("Macro F1 Score", "0.99")
        st.metric("Species Classified", "8")
        st.metric("Training Segments", "11,171")

        st.markdown("---")
        st.caption("Architecture: ResNet-34 + progressive unfreezing")
        st.caption("Data: Macaulay Library passive acoustic recordings")
        st.caption("Training: MPS (Apple Silicon), 50 epochs")

    st.markdown("---")

    st.subheader("Species Covered")
    cols = st.columns(4)
    species_list = list(SPECIES_INFO.keys())
    for i, col in enumerate(cols):
        with col:
            for j in [i, i + 4]:
                if j < len(species_list):
                    st.markdown(f"**{SPECIES_INFO[species_list[j]]['common']}**")
                    st.caption(SPECIES_INFO[species_list[j]]['sound'][:60] + "...")

    st.markdown("---")

    st.subheader("Try It")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Classify Audio**\nUpload a recording and get a species prediction with confidence score.")
    with col2:
        st.info("**Species Guide**\nLearn about the vocalizations, habitat, and conservation status of each species.")
    with col3:
        st.info("**Model Performance**\nConfusion matrix, per-species F1 scores, and training history.")

    st.caption("Use the sidebar to navigate between pages.")


def page_classify(model, idx_to_class):
    st.header("Classify Audio")
    st.write("Upload a WAV or MP3 recording — BioSonar will identify the whale species.")

    uploaded = st.file_uploader("drop audio file here", type=["wav", "mp3", "flac", "m4a"])

    if uploaded:
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("processing audio..."):
            img, err = audio_to_spectrogram(tmp_path)

        os.unlink(tmp_path)

        if err:
            st.error(f"couldn't process audio: {err}")
            return

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Mel Spectrogram")
            st.image(img, use_container_width=True)

        with col2:
            top_class, top_prob, all_p = predict(model, idx_to_class, img)
            info   = SPECIES_INFO.get(top_class, {})
            common = info.get("common", top_class)

            if top_prob < CONF_THRESHOLD:
                st.warning(f"Low confidence ({top_prob:.1%}) — prediction may be unreliable")
                st.write("BioSonar flags uncertain detections rather than forcing a guess.")
            else:
                st.success(f"**{common}**")
                st.metric("Confidence", f"{top_prob:.1%}")

            if info:
                st.markdown("---")
                st.markdown(f"**Sound:** {info['sound']}")
                st.markdown(f"**Habitat:** {info['habitat']}")
                st.markdown(f"**Conservation:** {info['conservation']}")

        st.subheader("All Species Probabilities")
        st.pyplot(prob_chart(all_p))


def page_species():
    st.header("Species Guide")
    st.write("The 8 whale species BioSonar was trained to identify.")

    selected = st.selectbox(
        "select species",
        options=list(SPECIES_INFO.keys()),
        format_func=lambda k: SPECIES_INFO[k]["common"]
    )

    info = SPECIES_INFO[selected]
    st.subheader(info["common"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Vocalizations:** {info['sound']}")
        st.markdown(f"**Habitat:** {info['habitat']}")
    with col2:
        st.markdown(f"**Conservation status:** {info['conservation']}")
        st.markdown(f"**Fun fact:** {info['fun_fact']}")

    if selected == "beaked_whale":
        st.info("Beaked whale has the fewest training samples (400 segments). "
                "Model confidence for this species tends to be lower.")


def page_performance():
    st.header("Model Performance")
    st.write("ResNet-34 trained on 11,171 mel-spectrogram segments across 8 species.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", "99.6%")
    col2.metric("Macro F1", "0.99")
    col3.metric("Beaked Whale F1", "0.98")

    st.markdown("---")

    if os.path.exists(LC_PATH):
        st.subheader("Training History")
        st.image(LC_PATH, use_container_width=True)
    else:
        st.warning("learning_curve.png not found - run model/evaluate_1.py first")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(CM_PATH):
            st.subheader("Confusion Matrix")
            st.image(CM_PATH, use_container_width=True)
        else:
            st.warning("confusion_matrix.png not found")
    with col2:
        if os.path.exists(F1_PATH):
            st.subheader("Per-Species F1")
            st.image(F1_PATH, use_container_width=True)
        else:
            st.warning("f1_scores.png not found")

    st.markdown("---")
    st.subheader("Limitations & Ethics")
    st.markdown("""
**Class imbalance** — beaked whale has ~400 training segments vs 2000+ for blue/fin whale.
Class weighting during training partially compensates, but confidence for beaked whale is lower.

**Geographic bias** — recordings sourced from Macaulay Library, which skews toward North Atlantic populations.
Model may underperform on Pacific or Southern Ocean recordings of the same species.

**Frequency ceiling** — beaked whale calls occur above 20kHz but recordings are capped at 11025Hz (Nyquist limit at 22050Hz SR).
The model learns lower-frequency secondary vocalizations rather than the primary echolocation clicks.

**Intended use** — BioSonar is a triage tool for marine biologists reviewing large volumes of passive acoustic data.
It should flag recordings for human review, not make final identifications autonomously.
    """)


def main():
    icon = logo_img() or "🌊"
    st.set_page_config(
        page_title="BioSonar",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .stApp { background-color: #0a0f1e; color: #e8f4f8; }
    .stMetric { background-color: #111827; border-radius: 8px; padding: 12px; }
    </style>
    """, unsafe_allow_html=True)

    b64 = logo_b64()
    if b64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{b64}" width="60" style="margin-bottom: 8px;">',
            unsafe_allow_html=True,
        )
    st.sidebar.title("BioSonar")
    st.sidebar.caption("whale species classifier from passive acoustic data")

    page = st.sidebar.radio(
        "navigate",
        ["Home", "Classify Audio", "Species Guide", "Model Performance"]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("ResNet-34 · 8 species · 99.6% test acc")
    st.sidebar.caption("trained on Macaulay Library recordings")

    model, idx_to_class = load_model()

    if model is None and page == "Classify Audio":
        st.error(f"model weights not found at {MODEL_PATH} or classes.json missing")
        st.stop()

    if page == "Home":
        page_home()
    elif page == "Classify Audio":
        page_classify(model, idx_to_class)
    elif page == "Species Guide":
        page_species()
    elif page == "Model Performance":
        page_performance()


if __name__ == "__main__":
    main()
