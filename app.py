from flask import Flask, render_template, request
from blur import blur_face

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    if request.method == "POST":
        # save uploaded img
        img = request.files["file"]
        img_path = f"img/{img.filename}"
        img.save(f"static/{img_path}")

        # blur img and save
        img_path = blur_face(img_path)
        
    return render_template("result.html", filename=img_path)

if __name__ == "__main__":
    app.run(debug=True)
