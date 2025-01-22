import requests

# Test API
port = "8000"
url = f"http://127.0.0.1:{port}/predict/"
images = [f"./static/sample/{name}.png" for name in ("Sneaker", "Trouser")]

for img in images:
    with open(img, "rb") as image_file:
        # { Field-name: File-name, File-object, File-type }
        files = {"file": (img, image_file, "image/png")}
        response = requests.post(url, files=files)
        resp_json = response.json()

    print("Status:", response.status_code)
    print("Response:", resp_json)

    assert response.status_code == 200
