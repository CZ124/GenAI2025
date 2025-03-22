from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import io
import base64
from PIL import Image

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for flashing messages

def adjust_hue(image, deficiency):
    """
    Convert an image to HSV, shift the hue based on the deficiency type,
    and then convert it back to BGR.
    """
    # Convert image from BGR to HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Choose a hue shift based on deficiency type
    if deficiency == "red-green":
        hue_shift = 20  # Example: shift hue by +20 for red-green deficiency
    elif deficiency == "yellow-blue":
        hue_shift = -20  # Example: shift hue by -20 for yellow-blue deficiency
    else:
        hue_shift = 0

    # Adjust the hue channel (OpenCV hue range: 0-179)
    hsv_img[:, :, 0] = (hsv_img[:, :, 0].astype(int) + hue_shift) % 180
    hsv_img[:, :, 0] = hsv_img[:, :, 0].astype(np.uint8)

    # Convert back to BGR color space
    corrected_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return corrected_img

def convert_image_to_base64(image):
    """
    Convert an OpenCV image (BGR) to a Base64 string for embedding in HTML.
    """
    # Convert from BGR to RGB for PIL compatibility
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Verify that an image file was uploaded
        if 'image' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        deficiency = request.form.get('deficiency')
        if not deficiency:
            flash("Please select a deficiency type")
            return redirect(request.url)

        # Read the uploaded image into a numpy array
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            flash("Error processing image")
            return redirect(request.url)

        # Adjust image hue based on the selected deficiency
        adjusted_img = adjust_hue(img, deficiency)
        # Convert the processed image to a Base64 string for display
        img_base64 = convert_image_to_base64(adjusted_img)
        return render_template('result.html', image_data=img_base64)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
