from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_cities,
    split_cities,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_cities,
                inputs=["initial_data_train_labels",
                        "initial_data_train_features"],
                outputs=["sj_train_features", "sj_train_labels",
                         "iq_train_features", "iq_train_labels"],
                name="split_cities_node",
            ),
            node(
                func=preprocess_cities,
                inputs=["sj_train_features", "iq_train_features",
                        "sj_train_labels", "iq_train_labels"],
                outputs=["sj_train", "iq_train"],

                name="preprocess_features_node",
            )

        ]
    )
