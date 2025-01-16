# Mlflow + Flask

- Dependencies: [conda.yaml](/ML-Ops/conda.yaml)
- Config: [MLproject](/ML-Ops/MLproject)
- Train model using Torch-Vision: [base.ipynb](/ML-Ops/base.ipynb)
  - FashionMNIST
  - MobileNet_v2
- Trace model on MLFlow: [model](/ML-Ops/model)
- Serve model on Flask: [app.py](/ML-Ops/app/app.py)
- Docker settings: [Dockerfile](/ML-Ops/app/Dockerfile)

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

## Flask

```bash
cd app
python app.py
```

## Docker

```bash
cd app
docker build -t flask-app:0.1 .
docker image ls -a
docker run -p 8080:5000 --name test flask-app:0.1
```

127.0.0.1:8080

### remove

```bash
docker rm test
```
