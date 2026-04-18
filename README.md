# **HY-World-2.0-Demo**

HY-World-2.0-Demo is a powerful, experimental 3D reconstruction and Gaussian Splatting suite powered by the Tencent HY-World-2.0 model (WorldMirror). Designed to bridge the gap between 2D media and immersive 3D environments, this application accepts uploaded images or videos and generates high-fidelity point clouds, accurate camera poses, depth maps, surface normals, and dense Gaussian splats. The suite features a deeply customized, interactive web interface built with custom HTML, CSS, and JavaScript. Users can visualize the reconstructed 3D scenes and splats directly within the browser using the integrated Rerun viewer, or they can download the generated `.glb` and `.ply` files for external rendering in tools like SuperSplat or PlayCanvas. Fully GPU-accelerated and optimized with Flash Attention, HY-World-2.0-Demo provides an advanced, streamlined environment for testing next-generation generative world models.

<img width="2420" height="1080" alt="0417PNG" src="https://github.com/user-attachments/assets/7766222d-e82f-40b6-b81a-d777c05ef643" />

### **Key Features**

* **Universal 3D Reconstruction:** Upload a series of images or a continuous video to automatically generate a complete 3D scene, including point clouds, depth maps, normal maps, and camera poses.
* **Gaussian Splat Generation:** Extracts and processes dense Gaussian splats from the input media, outputting them as viewable Rerun recordings or downloadable `.ply` files for use in external 3D engines.
* **Integrated Rerun 3D Viewer:** Features embedded, interactive Rerun viewports directly within the web interface, allowing users to orbit, pan, and inspect the generated 3D meshes and splats without leaving the browser.
* **Custom Web Interface:** Abandons standard UI blocks for a highly responsive, custom frontend design featuring a dark "Ubuntu Aubergine" or "Orange Red" theme. It includes drag-and-drop media zones, dynamic execution logs, and interactive image carousels for depth and normal maps.
* **Dual Execution Modes:** The application is accessible both as standard, native Gradio components (`app.py`) and in a fully custom Gradio Server Mode (`gradio_server_app.py`), depending on your deployment and UI customization needs.

---

### **Repository Structure**

```text
├── example_gradio/
│   ├── 1.mp4
│   └── 1.png
├── hyworldmirror/
│   ├── comm/
│   ├── models/
│   └── utils/
├── app.py
├── gradio_server_app.py
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run the HY-World-2.0-Demo locally, configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU with sufficient VRAM to handle 3D reconstruction pipelines.

**1. Install Pre-requirements**
Update pip to the required version before installing the main dependencies:
```bash
pip install pip>=23.0.0
```

**2. Install Core Requirements**
Install the necessary diffusion, machine learning, 3D processing, and web server libraries. Place these in a `requirements.txt` file and execute `pip install -r requirements.txt`. Note the specific versions required for `flash-attn` and `gsplat`.

```text
# --- Core DL Stack ---
flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
torch==2.4.0
torchvision==0.19.0

# --- API / Backend ---
fastapi
pydantic
requests

# --- Media / Processing ---
moviepy==1.0.3
opencv-python
pillow_heif

# --- Scientific / Math ---
numpy<2.0.0
scipy
matplotlib

# --- 3D / Vision ---
trimesh
open3d==0.18.0
plyfile
pycolmap==3.10.0

# --- ML / Utils ---
tqdm
omegaconf
einops
torchmetrics
jaxtyping
typeguard
colorspacious
safetensors
onnxruntime
kernels
uniception

# --- Gaussian Splatting ---
gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl

# --- ReRun ---
rerun-sdk
gradio_rerun
jax
termcolor

# --- Gradio ---
gradio==5.49.1  # For app.py
# gradio==6.12.0  # Optional: For gradio_server_app.py
```

### **Usage**

After setting up your environment and ensuring your dependencies are installed, you can launch the application.

To run the native Gradio application:
```bash
python app.py
```

To run the custom Gradio Server version (ensure you have Gradio 6.12.0 installed for this mode):
```bash
python gradio_server_app.py
```

The script will initialize the WorldMirror model into VRAM. Once ready, it will expose a local web server (typically at `http://127.0.0.1:7860/`). Open this address in your browser to access the interface. Upload a video or a batch of images, select your reconstruction options, and click "Reconstruct 3D Scene".

### **License and Source**

* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/HY-World-2.0-Demo.git](https://github.com/PRITHIVSAKTHIUR/HY-World-2.0-Demo.git)
