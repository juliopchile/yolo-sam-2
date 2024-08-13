# yolo-sam-2
This project integrates the Segment Anything 2 model for segmentation tasks over YOLO detections.

## Instalation and setup
Before using this project, you must install SAM 2. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`.

For this project to work correctly, you need a functional Python environment on a Linux system. Currently, Segment Anything 2 can only be installed on Linux, making it a requirement. If you're using Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

Note:
1. It's recommended to create a new Python environment via [Anaconda](https://www.anaconda.com/) for this installation and install PyTorch 2.3.1 (or higher) via `pip` following https://pytorch.org/.
2. The step above requires compiling a custom CUDA kernel with the `nvcc` compiler. If it isn't already available on your machine, please install the [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.
3. If you see a message like `Failed to build the SAM 2 CUDA extension` during installation, you can ignore it and still use SAM 2 (some post-processing functionality may be limited, but it doesn't affect the results in most cases).
4. For more information and clarification, please refer to the official [Segment-Anything-2 repository](https://github.com/facebookresearch/segment-anything-2/tree/main)

### Step 0) Setup the enviroment
```bash
conda create --name yolo-sam-2 python=3.10
conda activate yolo-sam-2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 1) Clone this repository
```bash
git clone https://github.com/juliopchile/yolo-sam-2.git
cd yolo-sam-2
```

### Step 2) Clone and succesfully install Meta's Segment Anything 2 repository
SAM 2 and all its dependencies will be installed inside this project repository and used as a package for importing the necessary libraries.
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

Due to a bug in the segment-anything-2 codebase, after installation, you need to run the command `python setup.py build_ext --inplace`.
```bash
python setup.py build_ext --inplace
```

### Step 3) Install other dependencies
This proyect uses `Ultralytics`
```bash
pip install Ultralytics
```
