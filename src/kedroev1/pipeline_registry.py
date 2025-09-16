# src/proyecto_ml/pipeline_registry.py
from __future__ import annotations

from kedro.pipeline import Pipeline
from kedroev1.pipelines.data_engineering.pipeline import create_pipeline as de_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    data_engineering = de_pipeline()
    return {
        "__default__": data_engineering,  # o combínalo con otros si ya existen
        "data_engineering": data_engineering,
    }
