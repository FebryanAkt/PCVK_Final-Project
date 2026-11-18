# MediaPipe Model File Mirrors

Alternative download mirrors for [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) `.task` model files.

The official model files are hosted on `storage.googleapis.com`, which is blocked or unreliable in some regions. This repo provides the same files via GitHub Releases so they can be downloaded from anywhere.

## Available Models

| File | Description | Precision | Size |
|------|-------------|-----------|------|
| `face_landmarker.task` | Face mesh detection (478 landmarks) | float16 | ~3.6 MB |
| `hand_landmarker.task` | Hand landmark detection (21 points) | float16 | ~7.5 MB |

## Download

Download from the [latest release](https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases) or use `wget`:

```bash
# Face Landmarker
wget https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/face_landmarker.task

# Hand Landmarker
wget https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/hand_landmarker.task
```

## Use in JavaScript

Replace the `modelAssetPath` in your MediaPipe setup:

```javascript
const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
  baseOptions: {
    modelAssetPath:
      "https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/face_landmarker.task",
    delegate: "GPU",
  },
  runningMode: "VIDEO",
});
```

## Official Sources

If `storage.googleapis.com` is accessible in your region, use the official URLs:

- [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

## Related

- [MediaPipe Face Mesh: All 478 Landmark Points](https://www.sanderdesnaijer.com/blog/mediapipe-face-mesh-landmarks) - Complete visual reference with interactive explorer
- [MediaPipe Hand & Face Tracking in JavaScript](https://www.sanderdesnaijer.com/blog/mediapipe-hand-face-tracking) - Step-by-step setup tutorial
- [Official MediaPipe documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

## Versioning

Each release (`v1`, `v2`, etc.) is a snapshot of the official models at that point in time. Older release URLs keep working indefinitely. If the official models have been updated and this mirror is behind, [open an issue](https://github.com/sanderdesnaijer/mediapipe-model-mirrors/issues).

## License

The MediaPipe model files are released by Google under the [Apache 2.0 License](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE). See the [LICENSE](LICENSE) file for details.
