# Retrograde Room

A 3D sound room in the browser: sounds are balls in a room, played forward, backward and flipped, with a live transcript
and a big 3D word on the stage. This is the stand-alone version, where the built-in recorder can use the microphone.

Open it: https://rezaalmiro.github.io/retrograde-room/

Everything runs in the browser: recordings and loaded files stay on the device, nothing is uploaded.

## Credits and licences of the bundled files
- Speech recognition models: Whisper (OpenAI, Apache-2.0; ONNX export by onnx-community) and Abdou/quran-asr-small (MIT),
  split into the `asr-w*.txt` and `asr-ctc.*.txt` parts; run with onnxruntime-web (Microsoft, MIT; `asr-ort.wasm`)
  and transformers.js (Hugging Face, Apache-2.0).
- Quran text in `asr-quran.json`: Tanzil Quran Text (tanzil.net), copied verbatim; its full copyright notice is inside the file.
- `sample-ozymandias.mp3`: "Ozymandias" by Percy Bysshe Shelley, read for LibriVox (public domain).
- 3D drawing: three.js (MIT), loaded from jsDelivr.
