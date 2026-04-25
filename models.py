from pydantic import BaseModel, Field
from typing import Literal, List, Optional


class NormalClassScore(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class AerialAnomalyResult(BaseModel):
    object_id: int
    bbox: Optional[BoundingBox] = None

    best_normal_label: str
    normality_score: float = Field(ge=0, le=1)
    anomaly_score: float = Field(ge=0, le=1)
    margin_score: float = Field(ge=0, le=1)

    top_normal_matches: List[NormalClassScore]
    margin_score: float = Field(ge=0, le=1)
    distance_to_known_class: Optional[float] = None 
    known_class_threshold: Optional[float] = None 

    status: Literal[
        "known_normal_object",
        "uncertain_normal_object",
        "unknown_aerial_object"
    ]

    risk_level: Literal["low", "medium", "high"]

    explanation: str


class SkyScanReport(BaseModel):
    image_name: str
    total_objects: int
    highest_anomaly_score: float = Field(ge=0, le=1)

    overall_status: Literal[
        "normal_sky",
        "uncertain_activity",
        "unknown_aerial_object_detected"
    ]

    objects: List[AerialAnomalyResult]

    recommendation: str