# GEM-HD
Gaze-based Early Mental Health Detection

## Overview
We use webcam-based eye-gaze trajectories to assess cognitive engagement and fatigue.
Our system estimates uncertainty, detects drift, and self-evaluates — producing a transparent model card.

## Folder Structure

## Team Roles
- Member 1 – Data & Feature Engineer
- Member 2 – ML & Uncertainty Lead
- Member 3 – Evaluation & Ops Engineer

## How to Run
1. `python src/feature_extraction/extract_features.py`
2. `python src/modeling/infer_uncertainty.py`
3. `python src/evaluation/eval_agent.py --run configs/run.yaml --judge configs/judge.yaml`

## Example Output
- `results/model_card.md`
- `results/drift_report.json`

## Purpose
- Cognitive Health
- Multimodal AI
- Tools for Early Detection
