# OpenMV MobileNetV2 INT8 Inference Script
# Works with mobilenetv2_224_int8_openmv.tflite (int8 I/O)

import sensor, image, time, tf, pyb

# === SETTINGS ===
MODEL_PATH = "/openmv/mobilenetv2_224_int8.tflite"  # Put this at root of OpenMV
LABELS = ["NO_THREAT", "THREAT"]
IMG_SIZE = 224

# === CAMERA INIT ===
sensor.reset()
sensor.set_pixformat(sensor.RGB565)      # OpenMV camera uses RGB565
sensor.set_framesize(sensor.QVGA)        # 320x240
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)              # Disable AGC for stable colors
sensor.set_auto_whitebal(False)          # Disable AWB

clock = time.clock()

# === LOAD MODEL ===
net = tf.load(MODEL_PATH, load_to_fb=True)  # Load into framebuffer (saves RAM)

while True:
    clock.tick()
    img = sensor.snapshot()

    # Resize to model's input size
    resized = img.copy(x_scale=IMG_SIZE / img.width(), y_scale=IMG_SIZE / img.height())

    # Run inference
    # No normalization! Model expects int8 values directly
    out = net.classify(resized)[0].output()

    # Find best label
    max_index = max(range(len(out)), key=lambda i: out[i])
    label = LABELS[max_index]

    # Convert int8 output (-128 to 127) to probability-like confidence
    confidence = (out[max_index] + 128) / 255.0

    # Debug output
    print("Prediction:", label, " Confidence:", confidence)

    # Draw label on image
    img.draw_string(0, 0, f"{label} ({confidence:.2f})", color=(255, 0, 0), scale=2)

    # LED alert for threat
    if label == "THREAT" and confidence > 0.8:
        pyb.LED(1).on()  # Red LED
    else:
        pyb.LED(1).off()

    # FPS info
    print("FPS:", clock.fps())
