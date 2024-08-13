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

### Step 1) Clone and succesfully install Meta's Segment Anything 2 repository
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

### Step 2) Download SAM2 weights
```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```
or individually from:

- [sam2_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt)
- [sam2_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)
- [sam2_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)
- [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)

### Step 3) Manually modify the a SAM2 function
Modify the `_load_img_as_tensor()` function inside `segment_anything_2/sam2/utils/misc.py`. This function is at the moment of the creation of this proyect, faulty, so it need to be changed to work properly.

Original:
```python
def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width
```

Modified:
```python
def _load_img_as_tensor(img_path, image_size):
    # Open and convert image to RGB
    img_pil = Image.open(img_path).convert("RGB")
    # Resize the image
    img_resized = img_pil.resize((image_size, image_size))
    # Convert the PIL image to a numpy array
    img_np = np.array(img_resized)
    # Ensure the array is of type np.uint8
    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)
    # Normalize the numpy array if it's uint8
    img_np = img_np / 255.0 if img_np.dtype == np.uint8 else img_np
    # Convert numpy array to a PyTorch tensor and permute dimensions
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img_tensor, video_height, video_width
```

### Step 4) Clone this repository
```bash
git clone https://github.com/juliopchile/yolo-sam-2.git
cd yolo-sam-2
```

### Step 5) Install other dependencies
This proyect uses Ultralytics and notebooks, alongside Matplotlib and OpenCV.
```bash
conda install ipykernel
pip install Ultralytics matplotlib opencv-python
```
