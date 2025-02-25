from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_colors(image_path, num_colors=8):
    img = Image.open(image_path)
    img = img.resize((200, 200))  # Resize to reduce computation
    img_data = np.array(img).reshape(-1, 3)  # Convert to (N,3) array

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(img_data)

    colors = kmeans.cluster_centers_.astype(int)
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(*color) for color in colors]

    return colors, hex_colors

def generate_color_palette(colors, output_path):
    img = Image.new("RGB", (400, 50), "white")
    draw = ImageDraw.Draw(img)

    block_size = 400 // len(colors)
    for i, color in enumerate(colors):
        draw.rectangle([i * block_size, 0, (i + 1) * block_size, 50], fill=tuple(color))

    img.save(output_path)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        palette_path = os.path.join(app.config['OUTPUT_FOLDER'], "color_palette.png")
        colors, hex_colors = extract_colors(filepath)
        generate_color_palette(colors, palette_path)

        return render_template("result.html", hex_colors=hex_colors, image_filename=filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
