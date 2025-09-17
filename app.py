
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from ultralytics import YOLO
import torch

app = Flask(__name__)

# -------------------------------
# Configuración GPU/CPU automática
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# -------------------------------
# Inicializa YOLO en GPU
# -------------------------------
model = YOLO("yolov5s.pt")  # modelo preentrenado

# -------------------------------
# Diccionario de frames de Unity
# -------------------------------
frames = {0: None, 1: None, 2: None, 3: None}

# -------------------------------
# Endpoint para recibir frames de Unity
# -------------------------------
@app.route('/upload_frame/<int:cam_id>', methods=['POST'])
def upload_frame(cam_id):
    file = request.files['image']  # coincide con el WWWForm de Unity
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    frames[cam_id] = img
    return 'OK', 200

# -------------------------------
# Generador de frames para HTML
# -------------------------------
def gen_frames(cam_id):
    while True:
        if frames[cam_id] is None:
            continue

        frame = frames[cam_id].copy()

        # -------------------
        # YOLO detección GPU
        # -------------------
        results = model(frame)  # <-- usa model(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                if name not in ["person", "car"]:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Codificar frame para HTML
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# -------------------------------
# Rutas Flask
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    return Response(gen_frames(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------------
# Ejecutar app
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
