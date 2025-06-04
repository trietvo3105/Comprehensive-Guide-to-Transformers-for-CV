# Setup Instructions for Vision Transformer Hands-on Practice

This guide provides detailed setup instructions for working with Vision Transformers (ViT) in different hardware environments. Whether you have access to a GPU or are working with just a CPU, these instructions will help you get started with the hands-on exercises.

## Option 1: Google Colab (Recommended for Users Without a GPU)

Google Colab provides free access to GPU resources, making it ideal for training and fine-tuning transformer models without local GPU hardware.

### Setting Up Google Colab

1. **Access Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Sign in with your Google account

2. **Enable GPU Acceleration**:
   - Click on "Runtime" in the menu
   - Select "Change runtime type"
   - Set "Hardware accelerator" to "GPU"
   - Click "Save"

3. **Verify GPU Access**:
   - Run the following code to confirm GPU availability:
   ```python
   import torch
   print("GPU available:", torch.cuda.is_available())
   print("GPU device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
   ```

4. **Install Required Libraries**:
   - Run the following cell to install the necessary packages:
   ```python
   !pip install transformers datasets torch torchvision matplotlib
   ```

5. **Import Libraries**:
   ```python
   import torch
   import torchvision
   from transformers import ViTFeatureExtractor, ViTForImageClassification
   from datasets import load_dataset
   import matplotlib.pyplot as plt
   ```

### Using Pre-made Notebooks

For convenience, we've prepared Colab notebooks that you can use directly:

1. **Fine-tuning ViT on CIFAR-10 with PyTorch Lightning**:
   - Open [this notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_PyTorch_Lightning.ipynb)
   - Make a copy to your Google Drive by clicking "File" > "Save a copy in Drive"

2. **Image Classification with Hugging Face Transformers**:
   - Open [this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)
   - Make a copy to your Google Drive by clicking "File" > "Save a copy in Drive"

## Option 2: Local Setup with CPU

If you prefer to work locally without GPU acceleration, follow these instructions to set up your environment.

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Basic familiarity with command line operations

### Step 1: Create a Virtual Environment

Creating a virtual environment helps manage dependencies for different projects.

**For Windows**:
```bash
# Create a new directory for your project
mkdir transformer_practice
cd transformer_practice

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

**For macOS/Linux**:
```bash
# Create a new directory for your project
mkdir transformer_practice
cd transformer_practice

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Step 2: Install Required Packages

Install the necessary libraries for working with Vision Transformers:

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Hugging Face libraries and other dependencies
pip install transformers datasets matplotlib jupyter
```

### Step 3: Create a Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook
```

This will open a browser window. Create a new Python notebook by clicking "New" > "Python 3".

### Step 4: Basic ViT Example for CPU

Copy and paste the following code into your notebook to verify your setup:

```python
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
from io import BytesIO

# Download a sample image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Load pre-trained ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

# Display the image
import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis('off')
plt.show()
```

## Option 3: Local Setup with GPU

If you have a compatible NVIDIA GPU, you can accelerate training and inference significantly.

### Prerequisites

- NVIDIA GPU with CUDA support
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) installed

### Step 1: Create a Virtual Environment

Follow the same steps as in Option 2 to create and activate a virtual environment.

### Step 2: Install PyTorch with CUDA Support

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face libraries and other dependencies
pip install transformers datasets matplotlib jupyter
```

### Step 3: Verify GPU Support

Create a new Jupyter notebook and run:

```python
import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Print GPU information
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
```

### Step 4: Run the ViT Example with GPU Support

Modify the example from Option 2 to use GPU acceleration:

```python
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
from io import BytesIO

# Download a sample image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Load pre-trained ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

# Display the image
import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis('off')
plt.show()
```

## Memory Optimization Tips for Transformer Models

When working with transformer models, especially on systems with limited resources:

1. **Reduce Batch Size**: Start with a small batch size (e.g., 4 or 8) and increase gradually if your system can handle it.

2. **Use Mixed Precision Training**: If using a GPU with Tensor Cores (NVIDIA Volta, Turing, or Ampere architecture), enable mixed precision training:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   # In training loop
   with autocast():
       outputs = model(**inputs)
       loss = outputs.loss
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Gradient Accumulation**: Update weights after accumulating gradients from multiple batches:
   ```python
   accumulation_steps = 4  # Update weights after this many batches
   
   # In training loop
   with autocast():
       outputs = model(**inputs)
       loss = outputs.loss / accumulation_steps
   
   scaler.scale(loss).backward()
   
   if (batch_idx + 1) % accumulation_steps == 0:
       scaler.step(optimizer)
       scaler.update()
       optimizer.zero_grad()
   ```

4. **Model Pruning or Quantization**: For inference, consider using quantized models:
   ```python
   from transformers import AutoModelForImageClassification
   
   # Load quantized model
   model = AutoModelForImageClassification.from_pretrained(
       'google/vit-base-patch16-224',
       quantization_config={"bits": 8}
   )
   ```

## Troubleshooting Common Issues

### Out of Memory Errors

If you encounter CUDA out of memory errors:
- Reduce batch size
- Use a smaller model variant (e.g., 'google/vit-base-patch16-224' instead of 'google/vit-large-patch16-224')
- Enable gradient accumulation
- Use mixed precision training

### Slow Training on CPU

If training is too slow on CPU:
- Use a smaller dataset for experimentation
- Reduce the number of training epochs
- Consider using Google Colab's free GPU resources

### Package Installation Issues

If you encounter issues installing packages:
- Ensure you're using the correct version of pip: `pip --version`
- Try installing packages one by one to identify problematic dependencies
- Check for compatibility between PyTorch, CUDA, and your GPU driver version

## Next Steps

After setting up your environment, proceed to the hands-on exercises in the next section. These exercises will guide you through:

1. Loading and preprocessing image data
2. Fine-tuning a pre-trained Vision Transformer
3. Evaluating model performance
4. Using the model for inference on new images

Remember that while training on CPU is possible, it will be significantly slower than using a GPU. For complex models or larger datasets, we strongly recommend using Google Colab's free GPU resources or a local GPU setup if available.
