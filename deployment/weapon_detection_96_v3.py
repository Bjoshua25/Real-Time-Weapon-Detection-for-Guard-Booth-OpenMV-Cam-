import sensor, image, time, tf, pyb

# === SETTINGS ===
MODEL_PATH = "/openmv/mobilenetv2_96_int8.tflite"  # Make sure name matches
LABELS = ["NO_THREAT", "THREAT"]
IMG_SIZE = 96

# === CAMERA INIT ===
sensor.reset()
sensor.set_pixformat(sensor.RGB565)  # Camera format
sensor.set_framesize(sensor.QVGA)    # 320x240
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)          # Disable AGC
sensor.set_auto_whitebal(False)      # Disable AWB

clock = time.clock()

# === LOAD MODEL ===
net = tf.load(MODEL_PATH, load_to_fb=True)

while True:
    clock.tick()
    img = sensor.snapshot()

    # Resize for model
    resized_img = img.copy(
        x_scale=IMG_SIZE / img.width(),
        y_scale=IMG_SIZE / img.height()
    )

    # Run inference (float32 output)
    out = net.classify(resized_img)[0].output()

    # Get best class
    max_index = max(range(len(out)), key=lambda i: out[i])
    label = LABELS[max_index]
    confidence_percent = int(out[max_index] * 100)  # Already 0.0–1.0

    print(f"Prediction: {label} ({confidence_percent}%)")

    # Draw label
    color = (255, 0, 0) if label == "THREAT" else (0, 255, 0)
    img.draw_string(0, 0, f"{label} ({confidence_percent}%)", color=color, scale=2)

    # LED alert
    if label == "THREAT" and confidence_percent > 60:
        pyb.LED(1).on()
    else:
        pyb.LED(1).off()

    print(f"FPS: {clock.fps():.2f}")
