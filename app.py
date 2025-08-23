from flask import Flask, request, render_template_string, url_for
import numpy as np
from tensorflow.keras.applications import MobileNetV2 # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports]
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = MobileNetV2(weights="imagenet")  # Load model

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Image Recognition</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #ffd6e8, #ffeaf4);
            background-attachment: fixed;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            min-height: 100vh;
            color: #444;
        }
        .container {
            background: rgba(255, 255, 255, 0.85);
            padding: 30px;
            margin-top: 50px;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            max-width: 500px;
            width: 90%;
            text-align: center;
            animation: fadeIn 0.5s ease-in-out;
        }
        h2 {
            margin-bottom: 20px;
            color: #b30059;
        }
        input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            background: transparent;
            border: 2px dashed #b30059;
            border-radius: 10px;
            color: #b30059;
            width: 90%;
            cursor: pointer;
        }
        input[type="submit"] {
            margin-top: 15px;
            padding: 12px 25px;
            background: linear-gradient(45deg, #ff85a1, #ffbfd2);
            border: none;
            border-radius: 30px;
            color: white;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        input[type="submit"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        img {
            max-width: 100%;
            border-radius: 15px;
            margin-top: 15px;
        }
        .prediction {
            text-align: left;
            margin-top: 20px;
        }
        .progress {
            background: #f2f2f2;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0 15px;
        }
        .progress-bar {
            height: 20px;
            text-align: right;
            padding-right: 8px;
            color: white;
            font-size: 12px;
            line-height: 20px;
            animation: growBar 1s ease-in-out;
            background-size: 200% 200%;
            background-position: right center;
        }
        .bar1 { background: linear-gradient(90deg, #ff4b2b, #ff416c); }
        .bar2 { background: linear-gradient(90deg, #36d1dc, #5b86e5); }
        .bar3 { background: linear-gradient(90deg, #11998e, #38ef7d); }
        .bar4 { background: linear-gradient(90deg, #f7971e, #ffd200); }
        .bar5 { background: linear-gradient(90deg, #c471ed, #f64f59); }

        @keyframes growBar {
            from { width: 0; }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>AI Image Recognition</h2>
        <form action="/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br>
            <input type="submit" value="Predict">
        </form>

        {% if img_url %}
            <h3>Uploaded Image:</h3>
            <img src="{{ img_url }}" alt="Uploaded Image">
        {% endif %}

        {% if preds %}
            <div class="prediction">
                <h3>Predictions:</h3>
                {% for label, score in preds %}
                    <strong>{{ label }}</strong> - {{ score }}%
                    <div class="progress">
                        <div class="progress-bar bar{{ loop.index }}" style="width: {{ score }}%;">
                            {{ score }}%
                        </div>
                    </div>
                {% endfor %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    preds = None
    img_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)
            img_url = url_for("static", filename=f"uploads/{file.filename}")

            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            pred = model.predict(x)
            decoded = decode_predictions(pred, top=3)[0]
            preds = [(label, f"{score*100:.2f}") for (_, label, score) in decoded]

    return render_template_string(HTML_PAGE, preds=preds, img_url=img_url)

if __name__ == "__main__":
    app.run(debug=True)
