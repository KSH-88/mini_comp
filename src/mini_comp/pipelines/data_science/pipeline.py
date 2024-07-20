from kedro.pipeline import Pipeline, node, pipeline

from .nodes import write_output_file, wrapper_split_data, wrapper_get_best_model, wrapper_plot_fitted_values, comp_mean_abs_error


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=wrapper_split_data,
                inputs=["sj_train", "iq_train"],
                outputs=["sj_train_data", "sj_test_data",
                         "iq_train_data", "iq_test_data"],
                name="split_data_node",
            ),
            node(
                func=wrapper_get_best_model,
                inputs=["sj_train_data", "sj_test_data",
                        "iq_train_data", "iq_test_data"],
                outputs=["sj_fitted_model", "iq_fitted_model"],
                name="get_best_model_node",
            ),
            node(
                func=wrapper_plot_fitted_values,
                inputs=["sj_train_data", "sj_fitted_model",
                        "iq_train_data", "iq_fitted_model"],
                outputs=['sj_figs', 'iq_figs'],
                name="plot_fitted_values_node",
            ),
            node(
                func=write_output_file,
                inputs=["sj_unseen_test", "iq_unseen_test",
                        "sj_fitted_model", "iq_fitted_model"],
                outputs="submission",
                name="write_output_file_node",
            ),

            node(
                func= comp_mean_abs_error,
                inputs=["sj_test_data", "iq_test_data",
                        "sj_fitted_model", "iq_fitted_model"],
                outputs=["mae_sj", "mae_iq"],
                name="comp_mean_abs_error_node",
            ),

        ]
    )
