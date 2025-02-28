# 📜 Deep Learning for Image Restoration

## 🎯 Overview
This repository contains an implementation of **Deep Image Prior (DIP)** for fundamental **image restoration tasks**, including:

✔ **Denoising** – Removing noise while preserving fine details.  
✔ **Super-Resolution** – Enhancing image quality by increasing resolution.  
✔ **Inpainting** – Restoring missing or corrupted parts of an image.

Deep Image Prior (DIP), introduced by **Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky**, demonstrates that a **randomly initialized CNN** can serve as a strong image prior *without any training data*! This approach challenges conventional learning-based priors and offers an alternative for image restoration tasks.

---

## 🚀 Features
- **Deep Image Prior (DIP)** implementation in PyTorch.
- Image **denoising** using DIP without external datasets.
- **Super-resolution** enhancement using DIP.
- **Image inpainting** to restore missing image regions.
- Support for **multiple architectures**: U-Net, ResNet, SkipNet.
- Automatic **early stopping** to prevent overfitting.

---

## 🔧 Installation
This project requires Python 3 and PyTorch. You can install the required dependencies using:

```bash
pip install torch torchvision numpy matplotlib scikit-image PIL
```

---

## 📌 Usage

### **1️⃣ Running the Notebook**
Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/deep-image-prior.git
cd deep-image-prior
```

Run the Jupyter Notebook:

```bash
jupyter notebook Deep_Image_Prior.ipynb
```

### **2️⃣ Image Denoising**
To denoise an image, load a noisy image and define a DIP model:

```python
from models.skip import skip

net = skip(
    num_input_channels=32, num_output_channels=3, 
    num_channels_down=[128]*5, num_channels_up=[128]*5, num_channels_skip=[4]*5, 
    upsample_mode='bilinear', need_sigmoid=True, need_bias=True, pad='reflection')
```

Then, train the model and optimize the noise reduction:

```python
optimize(optimizer, params, closure, learning_rate, num_iterations)
```

### **3️⃣ Super-Resolution**
For image super-resolution, define the DIP model:

```python
net = get_net(input_depth=32, NET_TYPE='skip', pad='reflection', upsample_mode='bilinear')
```

Train the model and optimize the high-resolution reconstruction:

```python
optimize(OPTIMIZER, params, closure, LR, num_iter)
```

### **4️⃣ Image Inpainting**
Load an image with missing regions and define a U-Net-like model:

```python
net = skip(
    num_input_channels=2, num_output_channels=3, 
    num_channels_down=[128]*5, num_channels_up=[128]*5, num_channels_skip=[0]*5, 
    upsample_mode='nearest', need_sigmoid=True, need_bias=True, pad='reflection')
```

Train the model to fill missing regions:

```python
optimize(OPTIMIZER, params, closure, LR, num_iter)
```

---

## 📊 Results & Discussion

### **1️⃣ Denoising Results**
DIP achieves a **PSNR of 30.9 dB**, competitive with traditional denoising methods like BM3D and DnCNN.

| **Method**    | **PSNR (dB)** |
|--------------|--------------|
| Bicubic      | 25.5         |
| BM3D         | 30.4         |
| DnCNN        | 31.2         |
| **DIP (Ours)** | **30.9**     |

### **2️⃣ Super-Resolution Results**
DIP outperforms bicubic interpolation but lags behind deep-learning-based SR models.

### **3️⃣ Inpainting Performance**
DIP successfully restores missing image regions without external data but struggles with large missing parts.

| **Method** | **Visual Quality** | **Computational Cost** |
|-----------|----------------|--------------------|
| Bicubic   | Poor          | Low               |
| PatchMatch | Decent        | Medium            |
| **DIP (Ours)** | **High**   | **Medium**        |
| GAN-based | **Very High** | **High**          |

### **4️⃣ Key Takeaways**
- ✅ **DIP is a powerful unsupervised method** when datasets are unavailable.
- ✅ **Early stopping prevents overfitting**, ensuring clean restorations.
- ✅ **Future work should explore hybrid models** to combine DIP with trained priors.

---

## 🛠️ Model Architectures

### **U-Net Based Skip Architecture**
- Encoder-Decoder structure with skip connections.
- Skip connections preserve fine-grained details.
- Used for **denoising, super-resolution, and inpainting**.

### **Alternative Architectures**
- **ResNet-based DIP** – Better for fine detail recovery.
- **Texture Networks** – Designed for texture restoration.

---

## 📌 Limitations & Future Work

🔴 **Limitations:**
- DIP is **computationally expensive** (per-image optimization).
- Overfitting to noise is possible **without early stopping**.
- Struggles with **large occlusions** in inpainting.

✅ **Potential Improvements:**
- **Hybrid DIP + Pretrained Models** – Combining DIP with deep priors.
- **GANs for Inpainting** – Adversarial training for better missing region hallucination.
- **DIP for Video Restoration** – Extending DIP for sequential image restoration.

---

## 📜 References
- **Deep Image Prior** - Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky, *CVPR 2018*.
- BM3D Denoising - Dabov et al.
- Super-Resolution - Dong et al. (SRCNN), Ledig et al. (SRGAN).

---

## 🤝 Contributing
Feel free to **fork** this repository, submit **pull requests**, or suggest **issues**!

---

## 📜 License
This project is released under the **MIT License**. See `LICENSE` for details.

---

## 📬 Contact
For questions, suggestions, or collaborations, feel free to reach out:
📧 **Email:** mohamed.badi@student-cs.fr
📌 **GitHub Issues:** [Open an Issue](https://github.com/yourusername/deep-image-prior/issues)
