## 🚀 Quick start

1. **Clone and enter the repo**

   ```bash
   git clone https://github.com/veerprasad05/dlib-face-recognition.git
   cd dlib-face-recognition
   ```

2. **Create a Python 3.9 virtual environment**

   ```bash
   # Linux / macOS
   python3.9 -m venv venv
   source venv/bin/activate

   # Windows (PowerShell)
   py -3.9 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   Python’s `venv` module cleanly isolates all project packages.  
   If PowerShell complains about execution policy, set it to `RemoteSigned` once:

   ```powershell
   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

---

## 📦 Dependencies

### 1. System packages

| Platform | Must-have tools | One-liner |
| -------- | --------------- | --------- |
| **Ubuntu 22.04 / Debian** | compiler tool-chain, CMake | `sudo apt update && sudo apt install build-essential cmake`|
| **macOS (Homebrew)** | Xcode CLT, CMake | `brew install cmake`|
| **Windows 10/11** | CMake ≥ 3.18, *Desktop development with C++* in **Visual Studio Build Tools 2022** | Download CMake ➡ cmake.org and Build Tools installer ➡ visualstudio.com, then tick *MSVC v143*, *Windows SDK* and *C++ CMake tools* during setup|

> **Why CMake / C++ tools?** `face_recognition` compiles dlib from source on Linux & macOS; Windows avoids this by using the wheel bundled in this repo, but the tools are still required for other native packages.

### 2. Python packages (cross-platform)

| Package | Purpose |
| ------- | ------- |
| `face_recognition` (automatically pulls `dlib`) | HOG/CNN face encodings |
| `opencv-python` | webcam I/O & drawing |
| `facenet-pytorch`  | fast MTCNN face detector |
| `torch`, `torchvision`, `torchaudio` | backend for **facenet-pytorch** |
| `numpy` / `pickle` (stdlib) | numeric ops & model export |

---

## 🛠 Platform-specific install commands

### 🔸 Linux / macOS

```bash
# in the activated venv
pip install --upgrade pip
pip install face_recognition opencv-python facenet-pytorch
# install PyTorch (CPU only); choose CUDA selector on pytorch.org if you have a GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

*If CMake is still reported missing, ensure `which cmake` returns the binary and that its version is ≥ 3.18.*

### 🔸 Windows

```powershell
# 1) Make sure the Build Tools & CMake are on PATH (re-open terminal after install)
# 2) Activate venv, then:
pip install --upgrade pip
pip install opencv-python facenet-pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# 3) Install dlib from the pre-built wheel shipped in this repo
pip install .\dlib-19.22.99-cp39-cp39-win_amd64.whl        # local file
# 4) Finally:
pip install face_recognition
```

Using the wheel sidesteps the long native build and MSVC errors many users hit.
---

## ▶️ Running the pipeline

| Step | Script | Purpose |
| ---- | ------ | ------- |
| **1. Collect faces** | `python 01_face_datasets.py` | Captures **60** aligned images per person via webcam and stores them in `dataset/`. |
| **2. Encode faces** | `python 02_face_encodings.py` | Builds `trainer/encodings.pkl` from all images. |
| **3. Real-time recognition** | `python 03_face_recognition.py` | Opens the webcam and overlays name + confidence on every detected face. |

---

## 🆘 Troubleshooting

* **`Building wheel for dlib … error` (Linux/macOS):** verify CMake is installed and a C++ compiler is available. See the canonical issue thread.  
* **`cl.exe not found` / **“Microsoft Visual C++ 14.x required”** (Windows): the Visual Studio Build Tools were not installed or not selected with *Desktop development with C++* 🔗citeturn4search2turn4search7.  
* **“CUDA version X not offered”** – CPU wheels work universally; for GPU, follow the selector on the PyTorch *Start Locally* page to pick the matching CUDA wheel.

---

## 📚 References

* CMake install guides [🔗](https://cmake.org/download/)  
* `face_recognition` install instructions [🔗](https://github.com/ageitgey/face_recognition)
* PyTorch installation matrix [🔗](https://pytorch.org/get-started/locally/)  
* OpenCV-Python wheels [🔗](https://pypi.org/project/opencv-python/)
* facenet-pytorch on PyPI [🔗](https://pypi.org/project/facenet-pytorch/)
* Python `venv` docs [🔗](https://docs.python.org/3/library/venv.html)  