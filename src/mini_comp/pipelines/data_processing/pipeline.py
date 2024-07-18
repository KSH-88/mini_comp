from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table, #TODO change this after Paras's merge
    load_data,
    split_cities,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=load_data,
            #     inputs=["sj_train_labels", "sj_train_features"],
            #     outputs=["sj_train_labels", "sj_train_features"],
            #     name="load_data_node",
            # ),

            node(
                func=split_cities,
                inputs=["initial_data_train_labels", "initial_data_train_features"],
                outputs=["sj_train_features", "sj_train_labels", "iq_train_features", "iq_train_labels"],
                name="split_cities_node",
            ),
            node(
                func=impute_features,
                inputs=["sj_train_features", "iq_train_features"],
                outputs=["sj_train_features", "iq_train_features"],
                name="impute_features_node",
            ),
            node(
                func=select_features,
                inputs=["sj_train_features", "iq_train_features"],
                outputs=["sj_train_features", "iq_train_features"],
                name="select_features_node",

            ),
            node(
                func=join_features_labels,
                inputs=["sj_train_features", "sj_train_labels"],
                outputs=["sj_train", "iq_train"],
                name="join_features_labels_node",
            ),
            
        ]
    )
