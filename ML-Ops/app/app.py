from flask import Flask, render_template, request
from model import convert_image, predict

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
        image = convert_image(f"static/{img_path}")
        label, prob = predict(image)

    return render_template(
        "result.html", filename=img_path, label=label, prob=f"{prob * 100:.2f}%"
    )


if __name__ == "__main__":
    # To make the server accessible from outside the container
    app.run(host="0.0.0.0", port=5000, debug=True)
