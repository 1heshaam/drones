# SkyGuard OpenSet

SkyGuard OpenSet is a normal-sky anomaly detection prototype.

Unlike conventional drone detection systems, this prototype does not train on drone images. Instead, it models normal aerial objects such as birds, aircraft, helicopters, balloons, kites, clouds, and visual artefacts.

If an aerial object does not strongly match the normal sky, it is flagged as an unknown aerial object for human review.

## Core idea

Most drone detection systems try to learn what drones look like.

SkyGuard reverses the problem:

> Instead of teaching the model what drones look like, we teach it what the normal sky looks like.

## Why this matters

Drone shapes change constantly, especially in military, improvised, or custom-built UAVs. Training directly on known drone shapes can cause detectors to overfit to yesterday's UAV designs.

SkyGuard focuses on the stable part of the problem: the normal aerial environment.

## Pydantic integration

Pydantic is used to validate every AI output through structured schemas:

- normality score
- anomaly score
- risk level
- top normal matches
- recommendation
- machine-readable JSON incident report

This makes the system auditable and ready for integration with dashboards, APIs, and security workflows.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py