# 🛡️ **Real-Time Weapon Detection for Guard Booth (OpenMV Cam)**

---

## 🚀 **1. Project Overview**

**Objective:**
To develop a lightweight, real-time embedded system using the **OpenMV Cam** that detects weapons in a guard booth by classifying visual input as:

* Human only
* Human with weapon
* Weapon only
* No threat

**Target Deployment:**
Military checkpoints, border posts, guard booths, or secure facilities

**Expected Behavior:**

* Continuously monitor video stream
* Classify scene from frame snapshot
* Trigger alerts for weapon-related classes
* Operate fully offline, on low power, without internet

---

## 📅 **2. Project Phases & Timeline**

| Phase                                 | Tasks                                                   | Timeline  |
| ------------------------------------- | ------------------------------------------------------- | --------- |
| **Phase 1: Planning & Setup**         | Define scope, install tools, OpenMV IDE, hardware setup | Day 1–2   |
| **Phase 2: Dataset Collection**       | Collect & label images for 4 classes                    | Day 3–7   |
| **Phase 3: Model Training**           | Train lightweight CNN with TensorFlow                   | Day 8–10  |
| **Phase 4: Convert & Deploy**         | Convert model to TFLite, upload to OpenMV               | Day 11–12 |
| **Phase 5: Testing & Validation**     | Live testing, tune thresholds, evaluate accuracy        | Day 13–14 |
| **Phase 6: Response Mechanisms**      | Add buzzer, LED, SD save, GSM alert (optional)          | Day 15–17 |
| **Phase 7: Final Integration & Demo** | Enclose in booth, power test, demo                      | Day 18–20 |

---

## 🗂️ **3. Dataset Plan**

You need a labeled dataset with **4 categories**:

| Class                   | Description                                 | Sample Count             |
| ----------------------- | ------------------------------------------- | ------------------------ |
| `0` – No Threat         | Empty booth, walls, random objects          | 300+                     |
| `1` – Human Only        | Soldiers, civilians, people without weapons | 300+                     |
| `2` – Weapon Only       | Guns/knives alone (table, floor)            | 300+                     |
| `3` – Human With Weapon | People holding weapons                      | 300–500 (priority class) |

### Notes:

* Resize all images to `96x96` px
* Convert to grayscale (or RGB if needed)
* Augment images (flip, blur, brightness) to boost training

---

## 🧠 **4. Model Design**

You’ll build a **TinyCNN** suitable for OpenMV:

* Input: 96x96 grayscale
* Layers: Conv → Pool → Conv → Pool → Dense → Softmax
* Output: 4 classes

Training tools:

* Python, TensorFlow/Keras
* Train on your laptop or Colab
* Convert to `.tflite` using TFLite Converter

---

## ⚙️ **5. Model Deployment (OpenMV)**

* Copy `.tflite` model to SD card or board flash
* Use MicroPython + OpenMV's `tf` module
* Script to:

  * Capture live frame
  * Run `.classify()` on image
  * Parse output to label
  * Trigger response (buzzer, print, LED, etc.)

---

## 🧪 **6. Response System**

Depending on detection result, trigger:

| Action                           | Hardware Needed                 |
| -------------------------------- | ------------------------------- |
| **Buzzer** or **LED alert**      | GPIO + piezo buzzer or LED      |
| **Save frame to SD**             | Use `img.save("/sd/frame.jpg")` |
| **GSM/SIM alert**                | Integrate ESP32/Sim800L module  |
| **Serial out to central system** | UART connection to main board   |

---

## 📊 **7. Performance Evaluation**

Metrics to monitor:

* Classification Accuracy (on test set)
* False Positives / Negatives
* Inference time (should be < 200ms)
* Power usage (battery operation?)

---

## 📦 **8. Hardware Setup**

| Component                      | Purpose               |
| ------------------------------ | --------------------- |
| **OpenMV Cam H7/H7 Plus**      | Vision + ML           |
| MicroSD card                   | Store model + images  |
| Piezo Buzzer / LEDs            | Alerts                |
| Power supply (5V)              | Standalone deployment |
| (Optional) WiFi / GSM Shield   | Remote alerts         |
| 3D-printed or wooden enclosure | Weather protection    |

---

## 📁 **9. Folder/Project Structure (Local + OpenMV)**

```bash
weapon-detection-guard-booth/
├── dataset/
│   ├── no_threat/
│   ├── human_only/
│   ├── weapon_only/
│   └── human_with_weapon/
├── models/
│   ├── trained_model.h5
│   └── model.tflite
├── scripts/
│   └── openmv_classify.py
├── deployment/
│   ├── openmv_script.py
│   └── alerts/
└── README.md
```

---

## 🧰 **10. Tools Needed**

| Tool                             | Use                         |
| -------------------------------- | --------------------------- |
| **OpenMV IDE**                   | Deployment, camera control  |
| **TensorFlow / Keras**           | Model training              |
| **Python 3.8+**                  | Environment                 |
| **Colab or local Jupyter**       | Training environment        |
| **Image tools (CVAT, LabelImg)** | Dataset labeling (optional) |


