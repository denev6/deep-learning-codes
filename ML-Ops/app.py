from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import mlflow

from model.mobilenet import transform, labels

logged_model = "runs:/ffb263113ee64d05928679ef5bcda173/MobileNet_model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    if request.method == "POST":
        # Save uploaded img
        img = request.files["file"]
        img_path = f"img/{img.filename}"
        img.save(f"static/{img_path}")

        # Classify
        image = Image.open(f"static/{img_path}").convert("L")
        input_tensors = transform(image)
        input_tensor = input_tensors.unsqueeze(0)
        input_array = input_tensor.cpu().numpy()

        output = loaded_model.predict(input_array)[0]
        predicted = output.argmax()
        label = labels[predicted]
        prob = softmax(output)
        prob = prob[predicted]

    return render_template(
        "result.html", filename=img_path, label=label, prob=f"{prob * 100:.2f}%"
    )


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    prob = exp_logits / np.sum(exp_logits)
    return prob


if __name__ == "__main__":
    app.run(debug=True)
