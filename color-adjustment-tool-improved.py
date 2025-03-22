import numpy as np
import cv2

class ColorBlindAdjustmentTool:
    def __init__(self, confusion_angle, confusion_index, total_error_score,
                 perfect_tes=16, max_tes=100, max_ci=4.0):
        """
        Initialize the tool with the user's color test results.

        Parameters:
        - confusion_angle: The angle in degrees representing the color confusion axis
        - confusion_index: The severity of the color blindness (1.0 = normal vision, up to ~4.0 severe)
        - total_error_score: Overall score from the D-15 test (16 = perfect, 40+ = severe)
        - perfect_tes: TES for perfect vision (default 16)
        - max_tes: Max expected TES for normalization (default 100)
        - max_ci: Max expected CI for normalization (default 4.0)
        """
        self.confusion_angle = confusion_angle
        self.confusion_index = confusion_index
        self.total_error_score = total_error_score
        self.perfect_tes = perfect_tes
        self.max_tes = max_tes
        self.max_ci = max_ci

        self.severity = self._calculate_severity()
        self.adjustment_strength = self._calculate_adjustment_strength()

    def _calculate_severity(self):
        # Clamp and normalize CI
        ci_clamped = min(max(self.confusion_index, 0.1), self.max_ci)
        ci_component = 1.0 - min(ci_clamped / self.max_ci, 1.0)

        # Clamp and normalize TES
        tes_clamped = max(self.total_error_score, self.perfect_tes)
        tes_component = min((tes_clamped - self.perfect_tes) / (self.max_tes - self.perfect_tes), 1.0)

        # Weighted combination
        severity = 0.7 * ci_component + 0.3 * tes_component
        return min(max(severity, 0), 1.0)

    def _calculate_adjustment_strength(self):
        base_strength = np.sqrt(self.severity)
        return 0.1 + 0.9 * base_strength

    def _calculate_confusion_vector(self):
        angle_rad = np.radians(self.confusion_angle)
        return np.array([0, np.cos(angle_rad), np.sin(angle_rad)])

    def adjust_image(self, image, method='lab'):
        if method == 'lab':
            return self._adjust_image_lab(image)
        else:
            raise ValueError("Only 'lab' method is supported in this version.")

    def _adjust_image_lab(self, image):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        confusion_vector = self._calculate_confusion_vector()

        perp_angle_rad = np.radians(self.confusion_angle + 90)
        enhancement_vector = np.array([0, np.cos(perp_angle_rad), np.sin(perp_angle_rad)])

        l_channel, a_channel, b_channel = cv2.split(lab_image)
        ab_channels = np.stack([a_channel, b_channel], axis=-1).astype(np.float32) - 128

        projection = np.zeros_like(l_channel, dtype=np.float32)
        for y in range(lab_image.shape[0]):
            for x in range(lab_image.shape[1]):
                ab_val = ab_channels[y, x]
                projection[y, x] = ab_val[0] * confusion_vector[1] + ab_val[1] * confusion_vector[2]

        projection_min = np.min(projection)
        projection_max = np.max(projection)
        if projection_max > projection_min:
            projection = (projection - projection_min) / (projection_max - projection_min) - 0.5

        a_adjusted = a_channel.astype(np.float32)
        b_adjusted = b_channel.astype(np.float32)
        l_adjusted = l_channel.astype(np.float32)

        luminance_enhancement = 0
        if self.total_error_score > self.perfect_tes:
            tes_ratio = min((self.total_error_score - self.perfect_tes) / 84.0, 1.0)
            luminance_enhancement = tes_ratio * 0.2

        for y in range(lab_image.shape[0]):
            for x in range(lab_image.shape[1]):
                adjust = self.adjustment_strength * projection[y, x] * 100
                a_adjusted[y, x] += adjust * enhancement_vector[1]
                b_adjusted[y, x] += adjust * enhancement_vector[2]
                if luminance_enhancement > 0:
                    l_diff = l_adjusted[y, x] - 128
                    l_adjusted[y, x] += l_diff * luminance_enhancement

        a_adjusted = np.clip(a_adjusted, 0, 255)
        b_adjusted = np.clip(b_adjusted, 0, 255)
        l_adjusted = np.clip(l_adjusted, 0, 255)

        adjusted_lab = cv2.merge([l_adjusted.astype(np.uint8),
                                  a_adjusted.astype(np.uint8),
                                  b_adjusted.astype(np.uint8)])
        return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

    def create_preview(self, image):
        adjusted = self.adjust_image(image, 'lab')
        h, w = image.shape[:2]
        preview = np.zeros((h, w*2, 3), dtype=np.uint8)
        preview[:, :w] = image
        preview[:, w:] = adjusted

        cv2.putText(preview, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(preview, "Adjusted", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(preview, f"Severity: {int(self.severity * 100)}%", (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        cv2.putText(preview, f"Angle: {self.confusion_angle:.1f}°", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        return preview

from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import tempfile
import webbrowser
import time



app = Flask(__name__)


def open_webpage():
    # URL for the webpage you want to open
    url = "http://127.0.0.1:5000"  # Update with the desired URL
    webbrowser.open(url)
    
# Optional: Add a small delay to ensure the browser opens properly.
time.sleep(2)
open_webpage()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/adjust-image', methods=['POST'])
def adjust_image():
    angle = float(request.form['confusion_angle'])
    index = float(request.form['confusion_index'])
    tes = float(request.form['total_error_score'])
    method = request.form.get('method', 'lab')

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    tool = ColorBlindAdjustmentTool(
        confusion_angle=angle,
        confusion_index=index,
        total_error_score=tes
    )

    adjusted_img = tool.adjust_image(img, method)

    # Save adjusted image to a temporary file and return it
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, adjusted_img)

    return send_file(temp_file.name, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)

    
