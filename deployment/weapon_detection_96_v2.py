# OpenMV MobileNetV2 INT8 Inference Script
# Works with mobilenetv2_96_int8_corrected.tflite (int8 I/O)

import sensor, image, time, tf, pyb

# === SETTINGS ===
MODEL_PATH = "/openmv/mobilenetv2_96_int8.tflite"  # Put this at root of OpenMV
LABELS = ["NO_THREAT", "THREAT"]
IMG_SIZE = 96

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
    confidence_percent = int(((out[max_index] + 128) / 255.0) * 100)

    # Debug output
    print("Prediction:", label, " Confidence:", confidence_percent)

    # Choose color based on label
    if label == "THREAT":
        color = (255, 0, 0) # Red
    else:
        color = (0, 255, 0) # Green

    # Draw label on image with percentage and color
    img.draw_string(0, 0, f"{label} ({confidence_percent}%)", color=color, scale=2)

    # FPS info
    print("FPS:", clock.fps())


