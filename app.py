import streamlit as st
from PIL import Image

from analyzer import analyse_image
from trained_analyzer import analyse_image_trained


st.set_page_config(
    page_title="SkyGuard OpenSet",
    page_icon="🛰️",
    layout="wide"
)


st.title("SkyGuard OpenSet")
st.subheader("Normal-Sky Anomaly Detection")

st.markdown(
    """
    This prototype does **not** train on drones.

    It models normal aerial categories such as birds, aircraft, helicopters, clouds, empty sky, and visual artefacts.
    If an object does not strongly match the normal sky, it is flagged as an **unknown aerial object**.
    """
)

st.info(
    "Core concept: we do not recognise drones directly. We reject objects that do not belong to the learned normal sky."
)


mode = st.radio(
    "Choose analysis mode",
    [
        "CLIP zero-shot baseline",
        "Trained normal-only classifier",
        "Hybrid comparison"
    ]
)


uploaded_file = st.file_uploader(
    "Upload a sky image",
    type=["jpg", "jpeg", "png"]
)


def show_report(report, title):
    st.write(f"## {title}")

    obj = report.objects[0]

    if obj.risk_level == "low":
        st.success("NORMAL SKY")
    elif obj.risk_level == "medium":
        st.warning("UNCERTAIN AERIAL OBJECT")
    else:
        st.error("UNKNOWN AERIAL OBJECT DETECTED")

    st.metric("Best normal match", obj.best_normal_label)
    st.metric("Normality score", f"{obj.normality_score:.2%}")
    st.metric("Class separation margin", f"{obj.margin_score:.2%}")
    st.metric("Anomaly score", f"{obj.anomaly_score:.2%}")
    st.metric("Risk level", obj.risk_level.upper())

    st.write("### Top normal matches")
    for match in obj.top_normal_matches:
        st.write(f"- {match.label}: **{match.score:.2%}**")

    st.write("### Explanation")
    st.write(obj.explanation)

    st.write("### Recommendation")
    st.write(report.recommendation)

    st.write("### Pydantic-validated JSON report")
    st.json(report.model_dump())


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if mode == "CLIP zero-shot baseline":
        with st.spinner("Running CLIP baseline..."):
            report = analyse_image(image, uploaded_file.name)

        show_report(report, "CLIP zero-shot baseline result")

    elif mode == "Trained normal-only classifier":
        with st.spinner("Running trained normal-only classifier..."):
            report = analyse_image_trained(image, uploaded_file.name)

        show_report(report, "Trained normal-only classifier result")

    else:
        col1, col2 = st.columns(2)

        with st.spinner("Running hybrid comparison..."):
            clip_report = analyse_image(image, uploaded_file.name)
            trained_report = analyse_image_trained(image, uploaded_file.name)

        with col1:
            show_report(clip_report, "CLIP baseline")

        with col2:
            show_report(trained_report, "Trained normal-only model")

        st.write("## Hybrid decision")

        clip_risk = clip_report.objects[0].risk_level
        trained_risk = trained_report.objects[0].risk_level

        if clip_risk == "high" or trained_risk == "high":
            st.error("HYBRID RESULT: UNKNOWN AERIAL OBJECT — REVIEW REQUIRED")
            st.write(
                "At least one model rejected the image as outside the normal sky. "
                "The system recommends human review."
            )
        elif clip_risk == "medium" or trained_risk == "medium":
            st.warning("HYBRID RESULT: UNCERTAIN — REVIEW RECOMMENDED")
            st.write(
                "The models did not fully agree or one model had moderate confidence."
            )
        else:
            st.success("HYBRID RESULT: NORMAL SKY")
            st.write(
                "Both models accepted the image as part of the normal sky."
            )

else:
    st.info("Upload an image to begin.")