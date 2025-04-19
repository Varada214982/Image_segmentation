from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from PIL import Image
from model_utils import load_model, preprocess_image, get_segmentation_mask, decode_segmentation, overlay_segmentation, get_class_legend

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    original_image = None
    overlay_image = None
    legend = None
    summary = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            original_image = filename

            image = Image.open(filepath).convert("RGB")
            input_tensor = preprocess_image(image)
            mask = get_segmentation_mask(model, input_tensor)
            overlay = overlay_segmentation(image, decode_segmentation(mask))
            overlay_filename = f"overlay_{filename}"
            overlay_path = os.path.join(UPLOAD_FOLDER, overlay_filename)
            overlay.save(overlay_path)
            overlay_image = overlay_filename
            legend = get_class_legend(mask)
            summary = ", ".join(label for label, _ in legend)

    return render_template('index.html',
                           original_image=original_image,
                           overlay_image=overlay_image,
                           legend=legend,
                           summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
