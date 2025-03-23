from flask import Flask, render_template_string, request, jsonify
import numpy as np
import math
import json
import random

app = Flask(__name__)

# =========================
# Define the cap data
# =========================
# The first cap is the fixed "pilot" cap.
cap_data = [
    {"CapNo": "pilot", "Munsell": "10B 5/6", "xC": 0.228, "yC": 0.254, "R": 93,  "G": 130, "B": 160},
    {"CapNo": "1",     "Munsell": "5B 5/4",    "xC": 0.235, "yC": 0.277, "R": 99,  "G": 130, "B": 143},
    {"CapNo": "2",     "Munsell": "10BG 5/4",  "xC": 0.247, "yC": 0.301, "R": 96,  "G": 132, "B": 137},
    {"CapNo": "3",     "Munsell": "5BG 5/4",   "xC": 0.254, "yC": 0.322, "R": 97,  "G": 133, "B": 128},
    {"CapNo": "4",     "Munsell": "10G 5/4",   "xC": 0.264, "yC": 0.346, "R": 99,  "G": 133, "B": 119},
    {"CapNo": "5",     "Munsell": "5G 5/4",    "xC": 0.278, "yC": 0.375, "R": 102, "G": 133, "B": 111},
    {"CapNo": "6",     "Munsell": "10GY 5/4",  "xC": 0.312, "yC": 0.397, "R": 109, "G": 132, "B": 98},
    {"CapNo": "7",     "Munsell": "5GY 5/4",   "xC": 0.350, "yC": 0.412, "R": 119, "G": 128, "B": 84},
    {"CapNo": "8",     "Munsell": "5Y 5/4",    "xC": 0.390, "yC": 0.406, "R": 134, "G": 122, "B": 76},
    {"CapNo": "9",     "Munsell": "10YR 5/4",  "xC": 0.407, "yC": 0.388, "R": 140, "G": 117, "B": 82},
    {"CapNo": "10",    "Munsell": "2.5YR 5/4", "xC": 0.412, "yC": 0.351, "R": 145, "G": 113, "B": 96},
    {"CapNo": "11",    "Munsell": "7.5R 5/4",  "xC": 0.397, "yC": 0.330, "R": 146, "G": 111, "B": 105},
    {"CapNo": "12",    "Munsell": "2.5R 5/4",  "xC": 0.376, "yC": 0.312, "R": 145, "G": 111, "B": 114},
    {"CapNo": "13",    "Munsell": "5RP 5/4",   "xC": 0.343, "yC": 0.293, "R": 141, "G": 112, "B": 125},
    {"CapNo": "14",    "Munsell": "10P 5/4",   "xC": 0.326, "yC": 0.276, "R": 136, "G": 114, "B": 135},
    {"CapNo": "15",    "Munsell": "5P 5/4",    "xC": 0.295, "yC": 0.261, "R": 129, "G": 117, "B": 143}
]

# The reference (correct) order is the order in which the data is provided.
reference_order = [cap["CapNo"] for cap in cap_data]

# ===================================================
# Helper Functions for Color Space Conversion & Metrics
# ===================================================
def xy_to_uv(x, y):
    """Convert CIE 1931 xy coordinates to CIE 1976 u′, v′."""
    denom = 3 + 12 * y - 2 * x
    u = (4 * x) / denom
    v = (9 * y) / denom
    return u, v

def get_uv_coordinates(order, cap_data):
    """Given an order (list of CapNo), return a list of (u, v) tuples."""
    cap_dict = {cap["CapNo"]: cap for cap in cap_data}
    uv_list = []
    for cap_no in order:
        cap = cap_dict[cap_no]
        u, v = xy_to_uv(cap["xC"], cap["yC"])
        uv_list.append((u, v))
    return uv_list

def compute_TES(uv_list):
    """Compute the Total Error Score (TES) as the sum of squared Euclidean distances between adjacent u′, v′ points."""
    tes = 0
    for i in range(len(uv_list) - 1):
        du = uv_list[i+1][0] - uv_list[i][0]
        dv = uv_list[i+1][1] - uv_list[i][1]
        tes += du**2 + dv**2
    return tes

def compute_confusion_angle(uv_list):
    """Compute the confusion angle using PCA on the ordered u′, v′ coordinates."""
    data = np.array(uv_list)
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    # Compute covariance matrix
    cov = np.cov(data_centered, rowvar=False)
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argmax(eigvals)
    principal_vector = eigvecs[:, idx]
    angle_rad = math.atan2(principal_vector[1], principal_vector[0])
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def compute_confusion_index(tes, tes_normal):
    """Compute the Confusion Index (CI) as the ratio of TES to TES_normal."""
    return tes / tes_normal

# Compute the "normal" TES from the reference (correct) order.
uv_reference = get_uv_coordinates(reference_order, cap_data)
TES_normal = compute_TES(uv_reference)

# =========================
# Flask Routes and Endpoints
# =========================
@app.route("/")
def index():
    # Separate the fixed pilot cap from the draggable caps.
    pilot = [cap for cap in cap_data if cap["CapNo"] == "pilot"][0]
    draggable_caps = [cap for cap in cap_data if cap["CapNo"] != "pilot"]
    # Shuffle the draggable caps so they don’t start in reference order.
    random.shuffle(draggable_caps)
    # Render the HTML template.
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>D-15 Color Vision Test</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    /* Container for all caps in a single row */
    #caps-container {
      display: flex;
      flex-direction: row;
      gap: 10px;
      border: 2px dashed #aaa;
      padding: 10px;
      min-height: 80px;
      align-items: center;
      overflow-x: auto;
    }
    .cap {
      width: 60px;
      height: 60px;
      border: 2px solid #333;
      text-align: center;
      line-height: 60px;
      font-weight: bold;
      color: #fff;
      cursor: move;
      user-select: none;
    }
    /* The pilot cap is fixed and not draggable */
    .cap.fixed {
      cursor: default;
    }
    /* Visual feedback for dragging */
    .dragging {
      opacity: 0.5;
    }
    .over {
      outline: 2px dashed #555;
    }
  </style>
</head>
<body>
  <h1>D-15 Color Vision Test</h1>
  <p>The pilot cap (at the very front) is fixed; arrange the remaining caps to follow the pilot cap in the order that best matches your color perception.</p>
  
  <!-- All caps are placed in a single container -->
  <div id="caps-container">
    <div id="pilot" class="cap fixed" data-capno="{{pilot.CapNo}}" style="background-color: rgb({{pilot.R}}, {{pilot.G}}, {{pilot.B}});">
      {{pilot.CapNo}}
    </div>
    {% for cap in draggable_caps %}
      <div class="cap" draggable="true" data-capno="{{ cap.CapNo }}" style="background-color: rgb({{cap.R}}, {{cap.G}}, {{cap.B}});">
        {{ cap.CapNo }}
      </div>
    {% endfor %}
  </div>
  
  <button id="submit-btn" style="margin-top: 20px;">Submit Arrangement</button>
  <div id="results" style="margin-top:20px;"></div>
  
  <script>
    // Set up drag and drop for only draggable caps.
    const container = document.getElementById('caps-container');
    let dragSrcEl = null;
    
    // Only attach event listeners to elements that are draggable.
    let draggableCaps = document.querySelectorAll('#caps-container .cap[draggable="true"]');
    draggableCaps.forEach(function(cap) {
      cap.addEventListener('dragstart', handleDragStart, false);
      cap.addEventListener('dragenter', handleDragEnter, false);
      cap.addEventListener('dragover', handleDragOver, false);
      cap.addEventListener('dragleave', handleDragLeave, false);
      cap.addEventListener('drop', handleDrop, false);
      cap.addEventListener('dragend', handleDragEnd, false);
    });
    
    function handleDragStart(e) {
      dragSrcEl = this;
      e.dataTransfer.effectAllowed = 'move';
      this.classList.add('dragging');
    }
    
    function handleDragOver(e) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      return false;
    }
    
    function handleDragEnter(e) {
      this.classList.add('over');
    }
    
    function handleDragLeave(e) {
      this.classList.remove('over');
    }
    
    function handleDrop(e) {
      e.stopPropagation();
      // Prevent dropping on the fixed pilot cap.
      if (this.id === "pilot") {
        container.insertBefore(dragSrcEl, this.nextSibling);
        return false;
      }
      if (dragSrcEl !== this) {
        let allCaps = Array.from(container.children);
        let srcIndex = allCaps.indexOf(dragSrcEl);
        let targetIndex = allCaps.indexOf(this);
        if (srcIndex < targetIndex) {
          container.insertBefore(dragSrcEl, this.nextSibling);
        } else {
          container.insertBefore(dragSrcEl, this);
        }
      }
      return false;
    }
    
    function handleDragEnd(e) {
      this.classList.remove('dragging');
      Array.from(container.children).forEach(function(item) {
        item.classList.remove('over');
      });
    }
    
    // On submit, collect the order of caps (as they appear in the container) and send to the server.
    document.getElementById('submit-btn').addEventListener('click', function() {
      let order = [];
      let capDivs = container.querySelectorAll('.cap');
      capDivs.forEach(function(div) {
        order.push(div.getAttribute('data-capno'));
      });
      
      fetch("/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ order: order })
      })
      .then(response => response.json())
      .then(data => {
        let resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = "<h3>Results:</h3>" +
          "<p>Total Error Score (TES): " + data.TES.toFixed(4) + "</p>" +
          "<p>Confusion Angle: " + data.confusion_angle.toFixed(2) + "°</p>" +
          "<p>Confusion Index: " + data.confusion_index.toFixed(4) + "</p>";
      });
    });
  </script>
</body>
</html>
    """, pilot=pilot, draggable_caps=draggable_caps)

@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()
    order = data.get("order")
    # Expecting 16 caps (pilot + 15 draggable)
    if not order or len(order) != len(cap_data):
        return jsonify({"error": "Invalid order length."}), 400
    
    # Get the u,v coordinates for the submitted order.
    uv_order = get_uv_coordinates(order, cap_data)
    tes = compute_TES(uv_order)
    confusion_angle = compute_confusion_angle(uv_order)
    confusion_index = compute_confusion_index(tes, TES_normal)
    
    results = {
        "TES": tes,
        "confusion_angle": confusion_angle,
        "confusion_index": confusion_index
    }
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
