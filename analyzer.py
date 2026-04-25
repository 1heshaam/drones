from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from models import (
    SkyScanReport,
    AerialAnomalyResult,
    NormalClassScore,
)


NORMAL_SKY_PROMPTS = [
    "a bird flying in the sky",
    "a flock of birds flying in the sky",
    "an airplane flying in the sky",
    "a passenger aircraft flying in the sky",
    "a helicopter flying in the sky",
    "a hot air balloon in the sky",
    "a kite flying in the sky",
    "a paraglider in the sky",
    "a glider aircraft in the sky",
    "clouds in the sky",
    "an insect close to the camera",
    "a blurry object in the sky",
    "camera noise or motion blur in the sky",
]


_model = None
_processor = None


def load_clip():
    global _model, _processor

    if _model is None or _processor is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model.eval()

    return _model, _processor


def classify_risk(normality_score: float):
    if normality_score >= 0.70:
        return (
            "known_normal_object",
            "low",
            "normal_sky",
            "The object strongly matches known normal aerial categories.",
            "No immediate action required."
        )

    elif normality_score >= 0.45:
        return (
            "uncertain_normal_object",
            "medium",
            "uncertain_activity",
            "The object partially matches normal aerial categories, but confidence is not high enough.",
            "Review recommended if this is a sensitive area."
        )

    else:
        return (
            "unknown_aerial_object",
            "high",
            "unknown_aerial_object_detected",
            "The object does not match the normal aerial categories with sufficient confidence.",
            "Flag for human review or secondary sensor confirmation."
        )


def analyse_image(image: Image.Image, image_name: str) -> SkyScanReport:
    model, processor = load_clip()

    image = image.convert("RGB")

    inputs = processor(
        text=NORMAL_SKY_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]

    scores = []
    for label, score in zip(NORMAL_SKY_PROMPTS, probs):
        scores.append(
            NormalClassScore(
                label=label,
                score=float(score)
            )
        )

    scores = sorted(scores, key=lambda x: x.score, reverse=True)

    best_match = scores[0]
    second_match = scores[1] if len(scores) > 1 else None

    normality_score = best_match.score
    second_score = second_match.score if second_match else 0.0
    margin_score = normality_score - second_score

    anomaly_score = 1 - normality_score

    status, risk_level, overall_status, explanation, recommendation = classify_risk(
        normality_score
    )

    result = AerialAnomalyResult(
        object_id=1,
        bbox=None,
        best_normal_label=best_match.label,
        normality_score=normality_score,
        anomaly_score=anomaly_score,
        margin_score=margin_score,
        top_normal_matches=scores[:3],
        status=status,
        risk_level=risk_level,
        explanation=explanation
    )

    report = SkyScanReport(
        image_name=image_name,
        total_objects=1,
        highest_anomaly_score=anomaly_score,
        overall_status=overall_status,
        objects=[result],
        recommendation=recommendation
    )

    return report