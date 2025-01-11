# Torch + Mlflow

- Load dataset and model from torch-vision
- Trace model on mlflow
- Serve model on mlflow

# Run Model

## run MLProject

```bash
mlflow run . -e main -P epoch=3 -P lr=0.005
mlflow run . -e predict -P img="./data/sample.png"
```

## open mlflow-ui

```bash
mlflow ui
```
