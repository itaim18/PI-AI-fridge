
# Smart Fridge Inventory System with Raspberry Pi 5

Automatically track what's inside your fridge using a Raspberry Pi 5, an ultrasonic sensor, and a custom AI camera model based on YOLOv8. Tired of discovering you’re out of eggs or milk only when you open the fridge at night? This project continuously monitors and updates you in real time—so you always know what's in stock.

## Demo
Check out the demo video in the repository (or linked below) to see the system in action:  
[**Demo Video**](assets\demo.mp4)

---

## Overview

- **Goal**: Detect when the fridge door opens/closes (via ultrasonic sensor), trigger an AI camera to recognize key products (milk, cheese, eggs, etc.), and broadcast a real-time inventory update to a React dashboard.
- **Motivation**: 
  - I was fed up with coming home after a busy day only to realize I was missing essential groceries. 
  - This system ensures I always know what needs restocking, even when I’m still at the store.

### Key Features
1. **Raspberry Pi 5** as the central processing unit.  
2. **YOLOv8n-based AI Camera** (exported to an IMX500-compatible model).  
3. **Ultrasonic Sensor** to detect door distance (activates camera when < 50cm).  
4. **Flask & SocketIO** for real-time communication.  
5. **React Dashboard** that displays “IN” (available) and “OUT” (missing) items.

---

## Hardware & Requirements

1. **Raspberry Pi 5** (with Python 3 installed).
2. **AI Camera Module** (compatible with YOLOv8/IMX500 exports).
3. **Ultrasonic Sensor** (HC-SR04 or similar).
4. **(Optional) PIR Motion Sensor** (HC-SR501 or similar).
5. **Jumper Wires** + Breadboard (for wiring sensors).
6. **Internet Connection** (for Flask server & SocketIO to broadcast data).

### Wiring Diagram
See the `docs/` or `assets/` folder for a Fritzing diagram illustrating the setup.  
![Fridge AI Fritzing](/assets/fridge_ai.jpg)

---

## Getting Started

1. **Clone this Repository**  
   ```bash
   git clone https://github.com/your-username/fridge-inventory-pi5.git
   cd fridge-inventory-pi5
   ```

2. **Install Dependencies**  
   - Make sure you have Python 3.x on your Raspberry Pi.  
   - Install Python packages (Flask, SocketIO, etc.):  
     ```bash
     pip install -r requirements.txt
     ```

3. **Model & Camera Setup**  
   - Place your YOLOv8n (IMX500 exported) model in the `models/` folder.  
   - Update the path in `detect_products.py` to point to your model file.
   - **Roboflow Dataset**: [https://app.roboflow.com/wow-7vmpw/ai-fridge-fssaf/1](https://app.roboflow.com/wow-7vmpw/ai-fridge-fssaf/1)

4. **Configure Sensor Pins**  
   - Check and update the GPIO pins in `ultrasonic_sensor.py` (and `pir_sensor.py` if using PIR).  
   - Refer to your Raspberry Pi pin layout to ensure accurate connections.

---

## Usage

### Running the Flask Server
1. Open a terminal on your Raspberry Pi (or any machine running Flask):
   ```bash
   cd fridge-inventory-pi5/server
   python app.py
   ```
   This will start a SocketIO-enabled Flask server on `http://0.0.0.0:5000`.

2. (Optional) Expose it to the internet using a service like **ngrok**:
   ```bash
   ngrok http 5000
   ```
   Copy the generated forwarding URL for use in your React app.

### Running the Sensor Loop
- On the Raspberry Pi, run the sensor loop script (ultrasonic or PIR-based). For example:
  ```bash
  python sensor_loop.py
  ```
  This script reads from the ultrasonic sensor, triggers camera captures, and sends data to the Flask server.

### React Dashboard
1. Navigate to the `dashboard` folder:
   ```bash
   cd fridge-inventory-pi5/dashboard
   ```
2. Install dependencies and start the React app:
   ```bash
   npm install
   npm start
   ```
3. Update the SocketIO URL in `src/App.js` (or wherever you initialize the socket) to either `http://<RPI-IP>:5000` or the ngrok URL.

---


## Project Structure

```
RPI5-FRIDGE-INVENTORY/
├── best_imx_model_fridge/
│   ├── best_imx_MemoryReport.json
│   ├── best_imx.onnx
│   ├── best_imx.pbtxt
│   ├── dnnParams.xml
│   ├── labels.txt
│   ├── network.rpk
│   └── packerOut.zip
│
├── server/
│   ├── fridge_server.ipynb
│   └── imx500_object_detection_fridge.py
│
└── README.md
```

- **`best_imx_model_fridge/`** – Contains the exported IMX model files (ONNX, PBTXT, labels, etc.) for your YOLOv8 to IMX500 conversion.  
- **`server/`** – Holds the main server logic.  
  - `fridge_server.ipynb` – A Jupyter notebook (possibly for testing or demonstration).  
  - `imx500_object_detection_fridge.py` – The Python script handling object detection with the IMX500 model.  
- **`README.md`** – Documentation for your project.



## Troubleshooting & Tips

- **Sensor Calibration**:  
  Experiment with the distance threshold (e.g., 50cm) in `sensor_loop.py` to get reliable door detection.
- **Camera Angle**:  
  Properly position the camera so it has a clear view of the fridge’s interior.
- **Performance**:  
  If inference is slow, consider using a more optimized version of YOLO or upgrading your Raspberry Pi’s resources.
- **Logging**:  
  Add `print` statements or use Python’s `logging` module to debug any real-time issues with sensors or model inference.

---

## References & Inspiration

- **Exporting YOLO to IMX500**: [YouTube Tutorial by LukeDitria](https://www.youtube.com/watch?v=I69lAtA2pP0&list=LL&index=1&t=34s&ab_channel=LukeDitria)
- **Roboflow Dataset**: [AI Fridge Dataset](https://app.roboflow.com/wow-7vmpw/ai-fridge-fssaf/1)
- **Full Blog Post**: [https://itaimiz.com/blog/rpi5-fridge](https://itaimiz.com/blog/rpi5-fridge)

---





**Enjoy a smarter fridge and never run out of essentials again!**
