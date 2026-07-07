# samples

`factory_people_demo.mp4` is used by CAM 01 for the local YOLOv8n video
detection demo.

Current source:

```text
intel-iot-devkit/sample-videos: worker-zone-detection.mp4
```

At runtime the Studio loads it through:

```text
/samples/factory_people_demo.mp4
```

If this file is absent, CAM 01 falls back to the browser canvas demo while the
same `/api/video-detections` payload continues to drive the overlay.
