# Kokoro Wyoming

This a [wyoming protocol](https://github.com/rhasspy/wyoming) server that implements [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) using the [Kokoro onnx](https://github.com/thewh1teagle/kokoro-onnx) python library.

Modified from [nordwestt/kokoro-wyoming](https://github.com/nordwestt/kokoro-wyoming) with some significant improvements

  - Updated to latest Kokoro model
  - Available voices are read directly from the model instead of hardcoding
  - Support for `SIGINT` and `SIGTERM` termination
  - Support for GPU accelerated inference (see [./docker-compose.intel.yml]())
  - Kokoro debug logging