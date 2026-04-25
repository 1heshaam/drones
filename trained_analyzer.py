from pathlib import Path
import json
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from models import (
    SkyScanReport,
    AerialAnomalyResult,
    NormalClassScore,
)


MODEL_PATH = Path("normal_sky_model.pt")
CLASS_NAMES_PATH = Path("class_names.json")
FEATURE_INDEX_PATH = Path("feature_index.pt")

_model = None
_class_names = None
_feature_index = None


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FeatureExtractor(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.features = nn.Sequential(*list(trained_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def load_feature_index():
    global _feature_index

    if _feature_index is not None:
        return _feature_index

    if not FEATURE_INDEX_PATH.exists():
        raise FileNotFoundError(
            "feature_index.pt not found. Run build_feature_index.py first."
        )

    _feature_index = torch.load(FEATURE_INDEX_PATH, map_location="cpu")
    return _feature_index


def load_trained_model():
    global _model, _class_names

    if _model is not None and _class_names is not None:
        return _model, _class_names

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "normal_sky_model.pt not found. Run train_classifier.py first."
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "class_names.json not found. Run train_classifier.py first."
        )

    with open(CLASS_NAMES_PATH, "r") as f:
        _class_names = json.load(f)

    device = get_device()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(_class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    _model = model
    return _model, _class_names


def classify_risk_open_set(normality_score: float, margin_score: float, best_label: str):
    high_confidence = normality_score >= 0.80
    medium_confidence = normality_score >= 0.60

    clear_margin = margin_score >= 0.25
    weak_margin = margin_score >= 0.15

    if high_confidence and clear_margin:
        return (
            "known_normal_object",
            "low",
            "normal_sky",
            f"The trained model confidently recognises this as '{best_label}' with a clear separation from other normal classes.",
            "No immediate action required."
        )

    elif medium_confidence and weak_margin:
        return (
            "uncertain_normal_object",
            "medium",
            "uncertain_activity",
            f"The model leans toward '{best_label}', but the confidence or class separation is not strong enough for a safe normal classification.",
            "Review recommended if this is a sensitive area."
        )

    else:
        return (
            "unknown_aerial_object",
            "high",
            "unknown_aerial_object_detected",
            f"The model's best match is '{best_label}', but confidence and separation are insufficient. This may be outside the learned normal sky.",
            "Flag for human review or secondary sensor confirmation."
        )


def analyse_image_trained(image: Image.Image, image_name: str) -> SkyScanReport:
    model, class_names = load_trained_model()
    device = get_device()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    feature_model = FeatureExtractor(model).to(device)
    feature_model.eval()

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

        feature = feature_model(tensor)
        feature = torch.nn.functional.normalize(feature, dim=1)[0].cpu()

    scores = []
    for label, score in zip(class_names, probs):
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

    feature_index = load_feature_index()

    centroid = feature_index["centroids"][best_match.label]
    threshold = feature_index["thresholds"][best_match.label]

    similarity_to_centroid = float(feature @ centroid)
    distance_to_centroid = 1 - similarity_to_centroid

    is_far_from_known_class = distance_to_centroid > threshold

    if is_far_from_known_class:
        status = "unknown_aerial_object"
        risk_level = "high"
        overall_status = "unknown_aerial_object_detected"
        explanation = (
            f"The model predicted '{best_match.label}', but the image is far from the learned "
            f"'{best_match.label}' feature cluster. This suggests it may be outside the normal sky."
        )
        recommendation = "Flag for human review or secondary sensor confirmation."
    else:
        status, risk_level, overall_status, explanation, recommendation = classify_risk_open_set(
            normality_score,
            margin_score,
            best_match.label
        )

    result = AerialAnomalyResult(
        object_id=1,
        bbox=None,
        best_normal_label=best_match.label,
        normality_score=normality_score,
        anomaly_score=anomaly_score,
        margin_score=margin_score,
        distance_to_known_class=distance_to_centroid,
        known_class_threshold=threshold,
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