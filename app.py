import json
import os
import tempfile

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from analyzer import analyse_image
from trained_analyzer import analyse_image_trained


st.set_page_config(
    page_title="SkyGuard OpenSet",
    page_icon="🛰️",
    layout="wide"
)


IMAGE_TYPES = ["jpg", "jpeg", "png"]
VIDEO_TYPES = ["mp4", "mov", "avi"]
ACCEPTED_TYPES = IMAGE_TYPES + VIDEO_TYPES

VIDEO_MAX_DURATION_SECONDS = 30
VIDEO_FRAME_SAMPLE_INTERVAL_SECONDS = 1


if "incidents" not in st.session_state:
    st.session_state["incidents"] = []


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
    "Upload a sky image or video",
    type=ACCEPTED_TYPES
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


def _file_extension(name: str) -> str:
    return os.path.splitext(name)[1].lower().lstrip(".")


def _is_video(name: str) -> bool:
    return _file_extension(name) in VIDEO_TYPES


def _save_uploaded_video(uploaded) -> str:
    suffix = "." + _file_extension(uploaded.name)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def _sample_video_frames(video_path: str):
    """Yield (timestamp_seconds, PIL.Image) for one frame per second up to the
    configured maximum duration. Skips silently when no frames can be read."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0 or fps != fps:  # NaN guard
        capture.release()
        return

    frame_step = max(int(round(fps * VIDEO_FRAME_SAMPLE_INTERVAL_SECONDS)), 1)
    max_frames = int(fps * VIDEO_MAX_DURATION_SECONDS)

    frame_index = 0
    try:
        while True:
            if max_frames and frame_index > max_frames:
                break

            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            timestamp = frame_index / fps
            yield timestamp, image

            frame_index += frame_step
    finally:
        capture.release()


def _row_from_report(timestamp_seconds: float, report) -> dict:
    obj = report.objects[0]
    return {
        "timestamp_seconds": round(float(timestamp_seconds), 2),
        "status": obj.status,
        "risk_level": obj.risk_level,
        "best_normal_label": obj.best_normal_label,
        "normality_score": float(obj.normality_score),
        "anomaly_score": float(obj.anomaly_score),
        "margin_score": float(obj.margin_score),
        "distance_to_known_class": (
            float(obj.distance_to_known_class)
            if obj.distance_to_known_class is not None else None
        ),
        "known_class_threshold": (
            float(obj.known_class_threshold)
            if obj.known_class_threshold is not None else None
        ),
        "recommendation": report.recommendation,
    }


def _analyse_video(uploaded) -> None:
    video_path = _save_uploaded_video(uploaded)

    st.video(video_path)

    rows = []
    incidents = []
    progress = st.progress(0.0, text="Analysing video frames...")

    samples = list(_sample_video_frames(video_path))

    if not samples:
        progress.empty()
        st.warning(
            "No readable frames were found in the uploaded video. "
            "The file may be corrupted or use an unsupported codec."
        )
        try:
            os.unlink(video_path)
        except OSError:
            pass
        return

    total = len(samples)
    for i, (timestamp, frame_image) in enumerate(samples, start=1):
        try:
            report = analyse_image_trained(
                frame_image,
                f"{uploaded.name}@{timestamp:.2f}s"
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to analyse frame at {timestamp:.2f}s: {exc}")
            progress.empty()
            try:
                os.unlink(video_path)
            except OSError:
                pass
            return

        row = _row_from_report(timestamp, report)
        rows.append(row)

        if row["risk_level"] in ("medium", "high"):
            incident = {
                "video_name": uploaded.name,
                **row,
            }
            incidents.append(incident)

        progress.progress(i / total, text=f"Analysed frame {i}/{total}")

    progress.empty()

    try:
        os.unlink(video_path)
    except OSError:
        pass

    df = pd.DataFrame(rows)

    st.write("## Per-frame analysis")
    st.dataframe(df, use_container_width=True)

    normal_count = sum(1 for r in rows if r["risk_level"] == "low")
    uncertain_count = sum(1 for r in rows if r["risk_level"] == "medium")
    unknown_count = sum(1 for r in rows if r["risk_level"] == "high")
    highest_anomaly = max((r["anomaly_score"] for r in rows), default=0.0)

    st.write("## Summary")
    summary_cols = st.columns(5)
    summary_cols[0].metric("Total frames", len(rows))
    summary_cols[1].metric("Normal", normal_count)
    summary_cols[2].metric("Uncertain", uncertain_count)
    summary_cols[3].metric("Unknown aerial", unknown_count)
    summary_cols[4].metric("Highest anomaly", f"{highest_anomaly:.2%}")

    st.session_state["incidents"].extend(incidents)

    st.write("## Incidents (medium / high risk)")
    if incidents:
        st.dataframe(pd.DataFrame(incidents), use_container_width=True)
    else:
        st.success("No medium or high risk frames in this video.")

    all_incidents = st.session_state["incidents"]
    st.download_button(
        label="Download incidents as JSON",
        data=json.dumps(all_incidents, indent=2),
        file_name="skyguard_incidents.json",
        mime="application/json",
        disabled=not all_incidents,
    )


def _analyse_image_upload(uploaded) -> None:
    image = Image.open(uploaded)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if mode == "CLIP zero-shot baseline":
        with st.spinner("Running CLIP baseline..."):
            report = analyse_image(image, uploaded.name)

        show_report(report, "CLIP zero-shot baseline result")

    elif mode == "Trained normal-only classifier":
        with st.spinner("Running trained normal-only classifier..."):
            report = analyse_image_trained(image, uploaded.name)

        show_report(report, "Trained normal-only classifier result")

    else:
        col1, col2 = st.columns(2)

        with st.spinner("Running hybrid comparison..."):
            clip_report = analyse_image(image, uploaded.name)
            trained_report = analyse_image_trained(image, uploaded.name)

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


if uploaded_file is not None:
    if _is_video(uploaded_file.name):
        st.caption(
            f"Video mode: sampling 1 frame every "
            f"{VIDEO_FRAME_SAMPLE_INTERVAL_SECONDS}s, "
            f"first {VIDEO_MAX_DURATION_SECONDS}s, "
            f"using the trained normal-only classifier."
        )
        _analyse_video(uploaded_file)
    else:
        _analyse_image_upload(uploaded_file)
else:
    st.info("Upload an image or a short video to begin.")
