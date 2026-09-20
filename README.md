# GMFSS ONNX DirectML

GMFSS_Fortuna is one of the strongest open models for anime/animation frame interpolation, combining optical flow (GMFlow) with a synthesis network tuned for hand-drawn and cel-shaded motion.

## [Download](https://github.com/fuyuka3725/gmfss-onnx-directml/releases)

Download Windows Executable for Intel/AMD/Nvidia GPU

**https://github.com/fuyuka3725/gmfss-onnx-directml/releases**

This package includes all the binaries and models required. It is portable, so no CUDA or PyTorch runtime environment is needed.

## Usages

**WARNING! : To use it, a minimum of 16GB RAM / 6GB VRAM (in FP16 mode) is required.**

Input video, or directory name containing two or more images, output interpolated video or image sequence.

### Example Commands

```cmd
gmfss-onnx-directml input.mp4 output.mp4 -x 2 -c --fp16
gmfss-onnx-directml input_dir output_dir -x 2 -c --fp16
```

### Full Usages

```console
usage: gmfss-onnx-directml.exe [-h] (-n TOTAL_FRAMES | -x MULTIPLIER) [-c] [--fp16] [--model-dir MODEL_DIR] [-S]
                               [--digits DIGITS]
                               input output

-x [factor]         Target total output frame count.
-n [pcs]            Interpolation multiplier applied uniformly to every frame pair.
-c                  Also duplicate the final frame to match the interpolation rate of the last pair.
--fp16              Use FP16 mode.
--model-dir         Folder containing the extracted models-v1.0 release. (default: models)

-S                  Use image sequence mode. (ex: gmfss-onnx-ml input_dir output_dir [argument...])
--digits [number]   Sequence mode only, output filename digit count. (default: 8)
```

## Build from Source

1. Install Python 3.10.11 or 3.11.

2. Creating a Virtual Environment `python -m venv .venv`, and settings `.venv\Scripts\activate.bat`

3. Install the required package:
python -m pip install --upgrade pip
pip install pyinstaller
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime-directml opencv-python numpy
pip install pillow

4. Test `.venv\Scripts\python.exe gmfss-onnx-directml.py input.mp4 output.mp4 [argument...]`

5. build `pyinstaller --onefile --clean --strip --collect-all onnxruntime --collect-all cv2 --collect-all PIL --add-data "driver/kernels/splat.cl;driver/kernels" gmfss-onnx-directml.py`

## Sample Images

### Original Image

<img width="540" height="320" alt="02" src="https://github.com/user-attachments/assets/61476b89-8ebe-452f-8757-f306be66676c" />

<img width="540" height="320" alt="01" src="https://github.com/user-attachments/assets/fc8b1a8f-3fc6-44d5-9b6e-36e483186b07" />

### Interpolate with gmfss

<img width="540" height="320" alt="gmfss" src="https://github.com/user-attachments/assets/462df687-a488-4dbd-9854-e5c76680c94b" />

```shell
gmfss-onnx-directml input_dir output_dir -x 2 -S --fp16
```

### Interpolate with rife-v4.25-heavy

<img width="540" height="320" alt="rife-v4 25-heavy" src="https://github.com/user-attachments/assets/52f7cad6-1c2d-4cd6-b356-df97bc914093" />

```shell
rife-ncnn-vulkan -0 input0 -1 input1 -m rife-v4.25-heavy -o output
```

## Credits

Origin: https://github.com/98mxr/GMFSS_Fortuna

ONNX Port: https://github.com/santiquiroz/port-gmfss-onnx
