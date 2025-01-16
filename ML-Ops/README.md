# Mlflow + Flask

- Dependencies: [conda.yaml](/ML-Ops/conda.yaml)
- Config: [MLproject](/ML-Ops/MLproject)
- Train model using Torch-Vision: [base.ipynb](/ML-Ops/base.ipynb)
  - FashionMNIST
  - MobileNet_v2
- Trace model on MLFlow
- Serve model on Flask: [app.py](/ML-Ops/app.py)

![mlflow-preview](/ML-Ops/preview/summary.png)

## MLFlow

### open MLFlow-ui

```bash
mlflow ui
```

### Train

```bash
mlflow run . -e main --experiment-name=MobileNetV2 -P epoch=3 -P lr=0.005
```

### Predict

```bash
mlflow run . -e predict -P img="./static/img/996.jpg"
```

### Serve

```bash
mlflow models serve -m "runs:/ffb263113ee64d05928679ef5bcda173/MobileNet_model" --port 8000
```

## Flask

```bash
python app.py
```
