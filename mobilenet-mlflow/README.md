# Torch + Mlflow

- Load dataset and model from torch-vision
- Trace and serve model on mlflow

## Run Model

### open mlflow-ui

```bash
mlflow ui
```

### Train

```bash
mlflow run . -e main --experiment-name=MobileNetV2 -P epoch=3 -P lr=0.005
```

### Predict

```bash
mlflow run . -e predict -P img="./data/sample.png"
```

### Serve

```bash
mlflow models serve -m "runs:/ffb263113ee64d05928679ef5bcda173/MobileNet_model" --port 8000
```
