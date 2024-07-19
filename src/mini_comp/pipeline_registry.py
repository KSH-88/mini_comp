"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data_processing import pipeline as data_processing_pipeline
from .pipelines.data_science import pipeline as data_science_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # return {
    #     "__default__": data_processing_pipeline + data_science_pipeline
    # }

    pipelines = find_pipelines()
    print("pipelines", pipelines)
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
