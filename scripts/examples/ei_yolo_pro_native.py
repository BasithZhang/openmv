# Basith - YOLO-Pro EON Object Detection on OpenMV / Arduino Nicla Vision
#
# This looks similar to the FOMO OpenMV example, but the actual YOLO-Pro EON
# inference runs natively in C++ inside the custom firmware through `ei_yolo`.

import sensor
import time
import ei_yolo

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)  # 320x240 live preview
sensor.skip_frames(time=2000)

# The model itself has a compiled 0.50 threshold.
# You can set this HIGHER here, but not lower than 0.50 without rebuilding model metadata.
min_confidence = 0.50

print("Native model:", ei_yolo.info())

clock = time.clock()
while True:
    clock.tick()
    img = sensor.snapshot()

    detections = ei_yolo.detect(img, min_confidence=min_confidence)

    for x, y, w, h, score, label in detections:
        img.draw_rectangle((x, y, w, h), color=(255, 0, 0), thickness=2)
        img.draw_string(x, max(0, y - 12), "%s %.2f" % (label, score), color=(255, 0, 0))
        print("%s score=%.3f x=%d y=%d w=%d h=%d" % (label, score, x, y, w, h))

    print("detections=%d fps=%.2f" % (len(detections), clock.fps()))
