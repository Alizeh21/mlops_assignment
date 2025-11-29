import kfp
from kfp import dsl
from src.pipeline_components import load_data, preprocess_data, train_model


@dsl.pipeline(
    name="Boston Housing MLOps Pipeline",
    description="A Kubeflow pipeline for training a regression model"
)
def ml_pipeline():
    step1 = load_data()
    step2 = preprocess_data()
    step3 = train_model(processed_data_path=step2.output)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=ml_pipeline,
        package_path="components/ml_pipeline.yaml"
    )
