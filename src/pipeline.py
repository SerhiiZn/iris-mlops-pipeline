from kfp import dsl
from kfp import compiler

@dsl.component(packages_to_install=['pandas', 'scikit-learn', 'kfp==2.17.0'])
def fetch_data(dataset_out: dsl.Output[dsl.Dataset]):
    import pandas as pd
    from sklearn.datasets import load_iris

    print("Fetching raw data...")
    iris = load_iris(as_frame=True)
    df = iris.frame

    df.to_csv(dataset_out.path, index=False)
    print(f"Data saved to {dataset_out.path}")

@dsl.component(packages_to_install=['pandas', 'scikit-learn', 'kfp==2.17.0'])
def preprocess_data(
    dataset_in: dsl.Input[dsl.Dataset],
    clean_dataset_out: dsl.Output[dsl.Dataset],
    scaler_type: str = "standard",
):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    print(f"Preprocessing data using {scaler_type} scaler...")
    df = pd.read_csv(dataset_in.path)

    X = df.drop(columns=["target"])
    y = df["target"]

    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
        
    X_scaled = scaler.fit_transform(X)
    clean_df = pd.DataFrame(X_scaled, columns=X.columns)
    clean_df["target"] = y.values

    clean_df.to_csv(clean_dataset_out.path, index=False)
    print("Preprocessing completed!")

@dsl.component(packages_to_install=['pandas', 'scikit-learn', 'joblib', 'kfp==2.17.0'])
def train_model(
    clean_dataset_in: dsl.Input[dsl.Dataset],
    model_out: dsl.Output[dsl.Model],
    c_parameter: float = 1.0,
) -> float:
    import os
    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(clean_dataset_in.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(C=c_parameter)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    print(f"Model trained. Accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(model_out.path), exist_ok=True)
    joblib.dump(clf, model_out.path)

    return acc

@dsl.pipeline(
    name="iris-classification-pipeline",
    description="MLOps pipeline for Iris classification"
)
def iris_pipeline(
    c_param: float = 1.0,
    scaler: str = "standard"
):
    fetch_task = fetch_data()
    
    preprocess_task = preprocess_data(
        dataset_in=fetch_task.outputs['dataset_out'],
        scaler_type=scaler
    )
    
    train_task = train_model(
        clean_dataset_in=preprocess_task.outputs['clean_dataset_out'],
        c_parameter=c_param
    )

if __name__ == "__main__":
    # Компиляция в YAML-файл при локальном или CI запуске
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path="iris_pipeline.yaml"
    )
    print("Pipeline successfully compiled to iris_pipeline.yaml !!!!")
