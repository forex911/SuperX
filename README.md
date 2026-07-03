# AI Image Enhancement Web Application

Transform your images with AI-powered 4× super-resolution enhancement using ESRGAN (Enhanced Super-Resolution Generative Adversarial Network).

![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- 🎨 **Modern Minimalist UI** - Elegant aesthetic design with clean typography
- 🚀 **4× Super-Resolution** - AI-powered upscaling using ESRGAN
- 📏 **Auto-Resize** - Automatically handles large images (up to 2000×2000px)
- 🧩 **Tile-Based Processing** - Memory-efficient processing for any image size
- 📱 **Fully Responsive** - Works seamlessly on desktop, tablet, and mobile
- ⚡ **Real-Time Preview** - See your image before enhancement
- 💾 **Easy Download** - One-click download of enhanced images
- 🎯 **Single Page Layout** - No scrolling, everything visible at once

## 🖼️ Demo

### Landing Page
Clean, elegant entrance with clear call-to-action

### Upload Interface
Drag & drop or click to upload images

### Results
Side-by-side comparison with download options

## 📋 Requirements

- Python 3.6 or higher
- PyTorch 1.9+
- OpenCV
- Flask 2.0+
- NumPy

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/sanjaiim25/image-enhancement-website.git

cd image-enhancement-website
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model

Download the pre-trained ESRGAN model and place it in the `models/` directory:
- **Model:** `RRDB_ESRGAN_x4.pth` (~64MB)
- **Location:** `models/RRDB_ESRGAN_x4.pth`

### 4. Run the Application

```bash
python app.py
```

### 5. Open in Browser

Visit `http://127.0.0.1:5000` in your web browser.

## 📂 Project Structure

```
image-enhancement-website/
├── app.py                      # Flask application (main backend)
├── RRDBNet_arch.py            # ESRGAN model architecture
├── requirements.txt           # Python dependencies
├── Procfile                   # Heroku deployment config
│
├── models/
│   ├── RRDB_ESRGAN_x4.pth    # Pre-trained model (not in git)
│   └── README.md              # Model information
│
├── static/
│   ├── landing.css            # Landing page styles
│   ├── index.css              # Upload page styles
│   ├── result.css             # Results page styles
│   ├── uploads/               # Temporary uploaded images
│   └── results/               # Enhanced output images
│
└── templates/
    ├── landing.html           # Landing page
    ├── index.html             # Upload interface
    └── result.html            # Results display
```

## 💡 Usage

### Basic Workflow

1. **Visit Landing Page** - Navigate to `http://127.0.0.1:5000`
2. **Click "Start Enhancing"** - Opens the upload interface
3. **Upload Image** - Drag & drop or click to browse
4. **Automatic Processing** - Image is enhanced automatically
5. **View Results** - See before/after comparison
6. **Download** - Click to download enhanced image

### Supported Formats

- **Input:** JPG, JPEG, PNG
- **Output:** PNG (lossless)
- **Max File Size:** 50MB
- **Max Dimensions:** Any size (auto-resized to 2000×2000 max)

### Image Size Guide

| Input Size | Auto-Resized | Output Size | Processing Time |
|------------|--------------|-------------|-----------------|
| 500×500 | (no resize) | 2000×2000 | ~30 sec |
| 1000×1000 | (no resize) | 4000×4000 | ~1-2 min |
| 1500×1500 | (no resize) | 6000×6000 | ~3-4 min |
| 2000×2000 | (no resize) | 8000×8000 | ~5-7 min |
| 4000×6000 | 1333×2000 | 5332×8000 | ~3-4 min |

**Note:** Large images are automatically resized to fit within 2000×2000 while maintaining aspect ratio.

## 🛠️ Technical Details

### AI Model

- **Architecture:** RRDB (Residual-in-Residual Dense Block)
- **Framework:** PyTorch
- **Model:** ESRGAN (Enhanced Super-Resolution GAN)
- **Scale Factor:** 4× upscaling
- **Training:** Pre-trained on DIV2K dataset

### Processing Method

**Tile-Based Algorithm:**
- Images split into 256×256 pixel tiles
- 16-pixel overlap for seamless blending
- Each tile processed independently
- Memory-efficient for large images
- Tiles stitched together automatically

**Device Support:**
- **GPU (CUDA):** Automatic detection and usage
- **CPU:** Fallback mode (slower but functional)

### Memory Optimization

- Processes one tile at a time
- Progressive memory release
- Row-by-row stitching
- Automatic garbage collection

## 🎨 Design Philosophy

### Aesthetic Minimalism

- **Typography:** Cormorant Garamond (serif) + Inter (sans-serif)
- **Colors:** Cream, sage, charcoal - natural muted palette
- **Layout:** Generous white space, clean lines
- **Animations:** Smooth, slow transitions
- **Inspiration:** Editorial design, zen aesthetics

### User Experience

- Single-screen layout (no scrolling)
- Instant visual feedback
- Clear call-to-actions
- Progressive disclosure
- Mobile-first responsive design

## ⚙️ Configuration

### Environment Variables (Optional)

```bash
FLASK_ENV=development          # or production
FLASK_DEBUG=1                  # Enable debug mode
SECRET_KEY=your-secret-key     # Flask secret key
```

### App Configuration

Edit `app.py` to customize:

```python
# Maximum file size (default: 50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Maximum input dimension (default: 2000px)
max_input_dimension = 2000

# Tile size for processing (default: 256px)
tile_size = 256

# Tile overlap (default: 16px)
overlap = 16
```

## 🚢 Deployment

### Heroku

1. Create a Heroku app:
```bash
heroku create your-app-name
```

2. Push to Heroku:
```bash
git push heroku main
```

3. Open your app:
```bash
heroku open
```

### Docker (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 🔧 Troubleshooting

### Common Issues

**"Out of memory" error:**
- Solution: Use smaller images (under 1500×1500)
- Close other applications
- Images auto-resize, but very large images may still exceed memory

**Slow processing:**
- Normal on CPU (GPU is much faster)
- Processing time scales with image size
- Expected: 30 sec to 7 min depending on size

**Image too large error:**
- Should auto-resize now
- If still occurs, try images under 2000×2000

**Model not found:**
- Ensure `RRDB_ESRGAN_x4.pth` is in `models/` directory
- Download from official source
- Check file size (~64MB)

## 📊 Performance

### CPU Mode
- Small images (500×500): 20-40 seconds
- Medium images (1000×1000): 1-2 minutes
- Large images (2000×2000): 5-7 minutes

### GPU Mode (CUDA)
- 10-20× faster than CPU
- Large images: 30-60 seconds
- Recommended for production

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ESRGAN Model:** [Enhanced Super-Resolution Generative Adversarial Networks](https://github.com/xinntao/ESRGAN)
- **PyTorch:** Deep learning framework
- **Flask:** Web framework
- **OpenCV:** Image processing library
- **Fonts:** Cormorant Garamond, Inter

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub

## 🗺️ Roadmap

### Planned Features
- [ ] Batch image processing
- [ ] Multiple AI models (REAL-ESRGAN, SwinIR)
- [ ] Image comparison slider
- [ ] Progress bar during processing
- [ ] User accounts and history
- [ ] API endpoint for programmatic access
- [ ] Docker support
- [ ] GPU acceleration guides
- [ ] Quality vs. speed presets
- [ ] Image format conversion

## 📸 Screenshots

### Landing Page
Beautiful minimalist entrance with clear value proposition.

### Upload Interface
Drag & drop interface with instant preview.

### Results Page
Side-by-side comparison with easy download.

---

**Made with ❤️ using AI and Python**

*Transform your images with the power of deep learning.*
