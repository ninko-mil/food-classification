import streamlit as st
from PIL import Image

from inference import FoodClassifier, CLASS_NAMES


st.set_page_config(
    page_title="Food-11 Classifier",
    page_icon="🍽️",
    layout="wide",
)


@st.cache_resource
def load_classifier():
    return FoodClassifier()


classifier = load_classifier()


st.markdown(
    """
<style>
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

.hero-box {
    padding: 1.8rem 2rem;
    border-radius: 20px;
    border: 1px solid rgba(128, 128, 128, 0.18);
    margin-bottom: 1.6rem;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}

.hero-text {
    font-size: 18px;
    color: #7b7b7b;
    max-width: 900px;
}

.prediction-card {
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid rgba(128, 128, 128, 0.18);
    margin-bottom: 1rem;
}

.prediction-label {
    color: #888;
    font-size: 14px;
    margin-bottom: 0.25rem;
}

.prediction-value {
    font-size: 38px;
    font-weight: 800;
    line-height: 1.1;
}

.metric-card {
    padding: 1rem;
    border-radius: 15px;
    border: 1px solid rgba(128, 128, 128, 0.16);
    text-align: center;
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
}

.metric-label {
    font-size: 13px;
    color: #888;
}

.section-space {
    margin-top: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="hero-box">
<div class="hero-title">🍽️ Food-11 Image Classifier</div>
<div class="hero-text">
Web aplikacija za klasifikaciju slika hrane korišćenjem treniranog ResNet18 modela.
Učitajte sliku i sistem će prikazati najverovatniju Food-11 klasu i nivo pouzdanosti.
</div>
</div>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Model")

    st.write("**Arhitektura:** ResNet18")
    st.write("**Broj klasa:** 11")
    st.write("**Ulaz:** 224 × 224 px")
    st.write("**Optimizer:** AdamW")
    st.write("**Learning rate:** 0.0001")

    st.divider()

    st.subheader("Food-11 klase")

    for i, class_name in enumerate(CLASS_NAMES, start=1):
        st.write(f"{i}. {class_name}")

    st.divider()

    st.caption(
        "Model je izabran na osnovu rezultata "
        "5-fold cross-validation eksperimenata."
    )


uploaded_file = st.file_uploader(
    "Učitajte sliku hrane",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file is None:
    st.info(
        "Izaberite JPG, JPEG ili PNG sliku kako biste pokrenuli klasifikaciju."
    )

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Model obrađuje sliku..."):
            result = classifier.predict(
                image=image,
                top_k=3,
            )

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        confidence_percent = confidence * 100

        image_col, result_col = st.columns(
            [1.05, 1],
            gap="large",
        )

        with image_col:
            st.subheader("Ulazna slika")

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True,
            )

        with result_col:
            st.subheader("Rezultat klasifikacije")

            st.markdown(
                f"""
<div class="prediction-card">
<div class="prediction-label">Predikovana klasa</div>
<div class="prediction-value">{predicted_class}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.write("**Pouzdanost predikcije**")

            st.progress(float(confidence))

            st.write(f"### {confidence_percent:.2f}%")

            st.metric(
                "Inference vreme",
                f"{result['inference_time_ms']:.2f} ms",
            )

            st.markdown(
                '<div class="section-space"></div>',
                unsafe_allow_html=True,
            )

            st.subheader("Top 3 predikcije")

            for index, prediction in enumerate(
                result["top_predictions"],
                start=1,
            ):
                class_name = prediction["class"]
                probability = prediction["probability"]
                probability_percent = probability * 100

                st.write(
                    f"**{index}. {class_name}** — "
                    f"{probability_percent:.2f}%"
                )

                st.progress(float(probability))

        st.divider()

        st.subheader("Informacije o modelu")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                """
<div class="metric-card">
<div class="metric-value">ResNet18</div>
<div class="metric-label">Arhitektura</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                """
<div class="metric-card">
<div class="metric-value">11</div>
<div class="metric-label">Klasa</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                """
<div class="metric-card">
<div class="metric-value">224 × 224</div>
<div class="metric-label">Ulaz</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with c4:
            st.markdown(
                """
<div class="metric-card">
<div class="metric-value">AdamW</div>
<div class="metric-label">Optimizer</div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.divider()

        st.warning(
            "Model je namenjen klasifikaciji slika hrane iz domena Food-11. "
            "Za slike van ovog domena može dati netačnu predikciju sa visokom "
            "verovatnoćom."
        )

    except Exception as exc:
        st.error(
            "Došlo je do greške prilikom obrade slike."
        )

        st.exception(exc)


st.divider()

st.caption(
    "Seminarski rad — Veštačka inteligencija sa primenama"
)