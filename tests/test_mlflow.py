import mlflow


def test_mlflow_logging():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("environment-test")

    with mlflow.start_run():
        mlflow.log_param("test_parameter", 1)
        mlflow.log_metric("test_metric", 0.5)