import os
from flask import Flask, render_template, request, jsonify, send_file
import ImageAnalyser
import uuid

app = Flask(__name__)

# Function to call YOLO model (replace this with your actual YOLO code)
def detect_objects(image):
    labels, annotated_img = ImageAnalyser.generate_segmented_image(image)
    return labels, annotated_img


@app.route('/')
def index():
    return render_template('index.html')

# /upload endpoint with the ability to call only to get labels or both labels and image
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Generate a unique filename using uuid
        unique_filename = str(uuid.uuid4()) + '.jpeg'
        image_path = os.path.join('static', unique_filename)

        # Save the uploaded image with the generated filename
        file.save(image_path)

        # Check if labels_only parameter is present in the request
        labels_only = request.args.get('labels_only', default=False, type=bool)

        # Call YOLO model function to get the labels
        labels, output_file = detect_objects(image_path)

        # Return only labels if labels_only is True
        if labels_only:
            return jsonify({'labels': labels})

        return jsonify({'labels': labels, 'image_path': output_file})
    else:
        return jsonify({'error': 'Method not allowed'}), 405


@app.route('/get_image')
def get_image():
    # Replace 'static/detected_image.jpg' with the actual path to your detected image
    
    return send_file('static/segmented_image.jpeg', mimetype='image/jpeg')

# if __name__ == '__main__':
#     # Use Gunicorn to run the app with multiple workers
#     # The '-w' flag specifies the number of worker processes
#     app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
