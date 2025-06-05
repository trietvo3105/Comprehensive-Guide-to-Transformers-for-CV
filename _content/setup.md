# Setup Instructions for Vision Transformer Hands-on Practice

This guide provides detailed instructions for setting up your environment to work with Vision Transformers. Whether you're using a local machine with or without a GPU, or a cloud-based solution like Google Colab, these instructions will help you get started quickly.

## Environment Options

You have several options for setting up your environment:

1. **Google Colab (Recommended for beginners)**: Free access to GPUs with minimal setup
2. **Local setup with GPU**: Fastest performance but requires compatible hardware
3. **Local setup without GPU**: Limited to smaller models but works on any computer
4. **Cloud-based alternatives**: Options like Kaggle Notebooks or AWS SageMaker

## Option 1: Google Colab Setup

Google Colab provides free access to GPUs and comes with many pre-installed libraries, making it ideal for beginners.

### Step 1: Access Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account

### Step 2: Create a New Notebook

1. Click on `File > New Notebook`
2. Rename the notebook by clicking on "Untitled0" at the top

### Step 3: Configure GPU Runtime

{% include figure.html
   src="https://miro.medium.com/v2/resize:fit:1400/1*Lad06lrjlU9UZgSTHUHLJA.png"
   alt="Google Colab GPU Setup"
   caption="Figure 1: Selecting GPU runtime in Google Colab"
%}

1. Click on `Runtime > Change runtime type`
2. Select `GPU` from the Hardware accelerator dropdown
3. Click `Save`

Free Colab sessions have limitations:

-   Sessions timeout after 12 hours of inactivity
-   Limited GPU usage per day
-   Shared resources may affect performance

### Step 4: Install Required Libraries

Run the following code in a cell to install the necessary libraries:

```python
!pip install torch torchvision tqdm matplotlib
!pip install timm  # For Vision Transformer implementations
```

### Step 5: Verify GPU Access

Run this code to confirm that PyTorch can access the GPU:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

## Option 2: Local Setup with GPU

If you have a compatible NVIDIA GPU, setting up a local environment will provide the best performance.

### Step 1: Install CUDA and cuDNN

1. Check your GPU compatibility at [NVIDIA's CUDA GPUs list](https://developer.nvidia.com/cuda-gpus)
2. Download and install the appropriate CUDA version from [NVIDIA's CUDA download page](https://developer.nvidia.com/cuda-downloads)
3. Download and install cuDNN from [NVIDIA's cuDNN page](https://developer.nvidia.com/cudnn) (requires free NVIDIA developer account)

### Step 2: Create a Conda Environment

1. Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open a terminal or Anaconda prompt
3. Create a new environment:

```bash
conda create -n vit python=3.8
conda activate vit
```

### Step 3: Install PyTorch with GPU Support

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Replace `cudatoolkit=11.3` with the version that matches your installed CUDA version. Check compatibility at [PyTorch's installation page](https://pytorch.org/get-started/locally/).

### Step 4: Install Additional Libraries

```bash
pip install timm matplotlib tqdm jupyter
```

### Step 5: Verify GPU Access

Launch Python and run:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

If you're having issues with GPU detection, try the following:

1. Ensure your GPU drivers are up to date
2. Check that CUDA and PyTorch versions are compatible
3. Try reinstalling PyTorch with the specific CUDA version you have installed

## Option 3: Local Setup without GPU

If you don't have a compatible GPU, you can still run smaller models on your CPU.

### Step 1: Create a Conda Environment

1. Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open a terminal or Anaconda prompt
3. Create a new environment:

```bash
conda create -n vit python=3.8
conda activate vit
```

### Step 2: Install PyTorch (CPU Version)

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Step 3: Install Additional Libraries

```bash
pip install timm matplotlib tqdm jupyter
```

Running Vision Transformers on CPU will be significantly slower than on GPU. Consider:

-   Using smaller models with fewer parameters
-   Reducing batch sizes
-   Processing fewer images
-   Using pre-computed features when possible

## Option 4: Cloud-based Alternatives

If Google Colab doesn't meet your needs, consider these alternatives:

### Kaggle Notebooks

1. Create an account at [Kaggle](https://www.kaggle.com/)
2. Go to "Notebooks" and click "New Notebook"
3. Under "Settings", select GPU accelerator
4. Libraries like PyTorch, torchvision, and timm are pre-installed

### AWS SageMaker

For more advanced users or those needing longer runtimes:

1. Create an [AWS account](https://aws.amazon.com/)
2. Navigate to SageMaker in the AWS console
3. Create a notebook instance with GPU support (e.g., ml.p3.2xlarge)
4. Choose a PyTorch or conda kernel

## Downloading Datasets

For the hands-on exercises, we'll use several datasets. Here's how to download them:

The main datasets we'll be using include:

1. **CIFAR-10**: A dataset of 60,000 32x32 color images in 10 classes
2. **Flowers-102**: A dataset of 102 flower categories
3. **ImageNet**: A subset for inference with pre-trained models

These datasets are automatically downloaded by the code in our exercises, but you can also pre-download them if you prefer.

### CIFAR-10

```python
import torchvision
# This will download CIFAR-10 to ./data/cifar-10
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
```

### ImageNet (Subset)

For exercises requiring ImageNet, we'll use a subset called ImageNette:

```python
!wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
!tar -xzf imagenette2-320.tgz
```

## Testing Your Environment

To ensure everything is set up correctly, run this simple test:

```python
import torch
import timm

# Check PyTorch and GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Test loading a ViT model
model = timm.create_model('vit_base_patch16_224', pretrained=True)
print(f"Model loaded successfully: {model.__class__.__name__}")

# Test moving model to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model moved to: {next(model.parameters()).device}")
```

If you see the model name printed and no errors, congratulations! Your environment is set up correctly and you're ready to start working with Vision Transformers.

## Troubleshooting Common Issues

### CUDA Out of Memory

-   Reduce batch size
-   Use a smaller model variant
-   Try gradient accumulation

### Package Conflicts

-   Create a fresh conda environment
-   Install packages in the recommended order
-   Check version compatibility between PyTorch and CUDA

### Import Errors

-   Ensure you've activated the correct environment
-   Reinstall problematic packages
-   Check for missing dependencies

### Slow Performance on GPU

-   Check if PyTorch is actually using the GPU (`next(model.parameters()).device`)
-   Update GPU drivers
-   Close other GPU-intensive applications

## Next Steps

Now that your environment is set up, you're ready to start working with Vision Transformers! Proceed to the Hands-on Practice section to begin implementing and experimenting with these powerful models.

In the hands-on practice, you'll learn how to:

-   Load and preprocess image data
-   Implement a basic Vision Transformer from scratch
-   Fine-tune pre-trained ViT models
-   Visualize attention maps
-   Apply ViT to various computer vision tasks
