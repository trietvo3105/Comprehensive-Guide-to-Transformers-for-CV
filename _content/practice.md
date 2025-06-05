# Hands-on Practice with Vision Transformers

This document provides step-by-step exercises and solutions for working with Vision Transformers (ViT) in computer vision tasks. These exercises are designed to be compatible with both Google Colab (recommended for users without a GPU) and local environments.

## Exercise 1: Image Classification with Pre-trained ViT

In this exercise, you'll use a pre-trained Vision Transformer model to classify images.

### Step 1: Setup Environment

```python
# Install required libraries
!pip install transformers torch torchvision matplotlib

# Import necessary libraries
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
```

### Step 2: Load Pre-trained Model

```python
# Load pre-trained ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")
```

### Step 3: Load and Process an Image

```python
# Function to load an image from URL
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Load a sample image
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image_from_url(image_url)

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')
plt.show()
```

### Step 4: Make Predictions

```python
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

# Get top 5 predictions
probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
top5_prob, top5_indices = torch.topk(probabilities, 5)

# Display top 5 predictions
for i, (prob, idx) in enumerate(zip(top5_prob, top5_indices)):
    print(f"#{i+1}: {model.config.id2label[idx.item()]} ({prob.item()*100:.2f}%)")
```

### Solution Analysis

This exercise demonstrates how to use a pre-trained ViT model for image classification. The model was trained on ImageNet and can recognize 1,000 different classes. The feature extractor handles all the necessary preprocessing, including resizing the image to 224x224 pixels and normalizing the pixel values.

The model's architecture divides the image into 16x16 patches, processes them through a transformer encoder, and uses the [CLS] token's output for classification. This approach allows the model to capture global relationships between different parts of the image.

In Vision Transformers, the attention mechanism computes the relationship between patches using the following equation:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V
$$

Where $Q$, $K$, and $V$ are the query, key, and value matrices derived from the image patches.

## Exercise 2: Fine-tuning ViT on a Custom Dataset

In this exercise, you'll fine-tune a pre-trained ViT model on the CIFAR-10 dataset.

### Step 1: Setup Environment

```python
# Install required libraries
!pip install transformers datasets torch torchvision matplotlib

# Import necessary libraries
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
```

### Step 2: Load and Prepare Dataset

```python
# Load CIFAR-10 dataset
dataset = load_dataset("cifar10")
print(dataset)

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Define image transformations
normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)

# Define function to preprocess images
def preprocess_train(examples):
    examples['pixel_values'] = [train_transforms(image.convert("RGB")) for image in examples['img']]
    examples['labels'] = examples['label']
    return examples

def preprocess_val(examples):
    examples['pixel_values'] = [val_transforms(image.convert("RGB")) for image in examples['img']]
    examples['labels'] = examples['label']
    return examples

# Apply preprocessing
train_dataset = dataset['train'].with_transform(preprocess_train)
test_dataset = dataset['test'].with_transform(preprocess_val)

# Create data loaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
```

### Step 3: Load Pre-trained Model for Fine-tuning

```python
# Load pre-trained model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=10,
    id2label={str(i): class_names[i] for i in range(10)},
    label2id={class_names[i]: str(i) for i in range(10)}
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")
```

### Step 4: Define Training Function

```python
# Define training function
def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Get inputs
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Define evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Get inputs
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values)

            # Get predictions
            _, predicted = torch.max(outputs.logits, 1)

            # Update statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
```

### Step 5: Train and Evaluate the Model

```python
# Set training parameters
num_epochs = 5
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training loop
for epoch in range(num_epochs):
    # Train
    train_loss = train(model, train_dataloader, optimizer, scheduler, device)

    # Evaluate
    train_accuracy = evaluate(model, train_dataloader, device)
    test_accuracy = evaluate(model, test_dataloader, device)

    # Print statistics
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
```

### Step 6: Save the Fine-tuned Model

```python
# Save the model
model.save_pretrained("./vit-cifar10")
feature_extractor.save_pretrained("./vit-cifar10")
print("Model saved to ./vit-cifar10")
```

### Step 7: Visualize Predictions

```python
# Function to visualize predictions
def visualize_predictions(model, dataset, feature_extractor, device, num_images=5):
    model.eval()
    fig, axes = plt.subplots(1, num_images, figsize=(20, 4))

    for i in range(num_images):
        # Get a random image
        idx = np.random.randint(0, len(dataset))
        image = dataset[idx]['img'].convert("RGB")
        label = dataset[idx]['label']

        # Prepare image for the model
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get predicted class
        predicted_class_idx = logits.argmax(-1).item()

        # Display image and prediction
        axes[i].imshow(image)
        axes[i].set_title(f"True: {class_names[label]}\nPred: {class_names[predicted_class_idx]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Visualize some predictions
visualize_predictions(model, dataset['test'], feature_extractor, device)
```

### Solution Analysis

This exercise demonstrates how to fine-tune a pre-trained ViT model on a custom dataset (CIFAR-10). The key steps include:

1. **Data Preparation**: Transforming images to the format expected by the ViT model (224x224 pixels) and applying data augmentation to improve generalization.

2. **Model Adaptation**: Modifying the classification head of the pre-trained model to output 10 classes instead of the original 1,000 ImageNet classes.

3. **Fine-tuning Strategy**: Using a small learning rate (5e-5) to update the model parameters without drastically changing the pre-trained weights.

4. **Evaluation**: Monitoring both training and test accuracy to ensure the model is learning effectively without overfitting.

The fine-tuned model should achieve around 85-90% accuracy on CIFAR-10 after just a few epochs, demonstrating the power of transfer learning with pre-trained Vision Transformers.

## Exercise 3: Attention Visualization in Vision Transformers

In this exercise, you'll visualize the attention patterns in a Vision Transformer to understand what the model is focusing on when making predictions.

### Step 1: Setup Environment

```python
# Install required libraries
!pip install transformers torch torchvision matplotlib numpy

# Import necessary libraries
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
```

### Step 2: Load Pre-trained Model and Image

```python
# Load pre-trained ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', output_attentions=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Load a sample image
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')
plt.show()
```

### Step 3: Extract Attention Maps

```python
# Prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Get model outputs including attention maps
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# Get prediction
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

# Extract attention maps
attention_maps = outputs.attentions  # This is a tuple of tensors

# Print attention map shapes
print("Number of layers:", len(attention_maps))
print("Attention map shape for first layer:", attention_maps[0].shape)
```

### Step 4: Visualize Attention Maps

```python
def visualize_attention(image, attention_maps, layer_idx=11, head_idx=0):
    """
    Visualize attention for a specific layer and attention head.

    Args:
        image: PIL Image
        attention_maps: Tuple of attention tensors from model output
        layer_idx: Index of the transformer layer to visualize
        head_idx: Index of the attention head to visualize
    """
    # Get attention map for specified layer and head
    attention = attention_maps[layer_idx][0, head_idx].detach().cpu().numpy()

    # We need to exclude the attention to the CLS token
    attention = attention[0, 1:]  # Shape: (num_patches)

    # Reshape attention to match image patches
    num_patches = int(np.sqrt(attention.shape[0]))
    attention_map = attention.reshape(num_patches, num_patches)

    # Resize image to match attention map visualization
    resized_image = image.resize((224, 224))

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Plot original image
    ax1.imshow(resized_image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Plot attention map
    ax2.imshow(attention_map, cmap='viridis')
    ax2.set_title(f"Attention Map (Layer {layer_idx+1}, Head {head_idx+1})")
    ax2.axis('off')

    # Plot overlay
    ax3.imshow(resized_image)
    ax3.imshow(attention_map, alpha=0.5, cmap='viridis')
    ax3.set_title("Attention Overlay")
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

# Visualize attention for the last layer, first head
visualize_attention(image, attention_maps, layer_idx=11, head_idx=0)

# Visualize attention for the last layer, different head
visualize_attention(image, attention_maps, layer_idx=11, head_idx=5)

# Visualize attention for an earlier layer
visualize_attention(image, attention_maps, layer_idx=5, head_idx=0)
```

### Step 5: Visualize Attention Across All Heads

```python
def visualize_all_heads(image, attention_maps, layer_idx=11):
    """
    Visualize attention for all heads in a specific layer.

    Args:
        image: PIL Image
        attention_maps: Tuple of attention tensors from model output
        layer_idx: Index of the transformer layer to visualize
    """
    # Get attention maps for specified layer
    attention = attention_maps[layer_idx][0].detach().cpu().numpy()

    # Number of attention heads
    num_heads = attention.shape[0]

    # Create figure
    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    axes = axes.flatten()

    # Plot attention for each head
    for head_idx in range(min(num_heads, 12)):
        # Get attention map for this head (excluding CLS token)
        head_attention = attention[head_idx, 0, 1:]

        # Reshape attention to match image patches
        num_patches = int(np.sqrt(head_attention.shape[0]))
        attention_map = head_attention.reshape(num_patches, num_patches)

        # Plot
        axes[head_idx].imshow(image.resize((224, 224)))
        axes[head_idx].imshow(attention_map, alpha=0.5, cmap='viridis')
        axes[head_idx].set_title(f"Head {head_idx+1}")
        axes[head_idx].axis('off')

    plt.suptitle(f"Attention Maps for Layer {layer_idx+1}", fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualize all heads for the last layer
visualize_all_heads(image, attention_maps, layer_idx=11)
```

### Solution Analysis

This exercise demonstrates how to visualize attention patterns in Vision Transformers, providing insights into what the model focuses on when making predictions. Key observations include:

1. **Different Attention Patterns**: Each attention head learns to focus on different aspects of the image. Some heads might attend to object shapes, while others focus on textures or colors.

2. **Layer Progression**: Earlier layers tend to capture more local features, while deeper layers develop more global attention patterns that correspond to semantic concepts.

3. **Interpretability**: Attention visualizations can help interpret the model's decision-making process, showing which parts of the image influenced the classification the most.

These visualizations reveal that Vision Transformers, unlike CNNs, can directly model long-range dependencies in images through their self-attention mechanism, allowing them to capture global context more effectively.

## Exercise 4: Transfer Learning with ViT for Custom Image Classification

In this exercise, you'll apply a pre-trained ViT model to a custom image classification task using transfer learning.

### Step 1: Setup Environment

```python
# Install required libraries
!pip install transformers torch torchvision matplotlib datasets

# Import necessary libraries
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
```

### Step 2: Load and Prepare a Custom Dataset

```python
# Load the Flowers dataset
dataset = load_dataset("huggan/flowers-102-categories")
print(dataset)

# Get class names
class_names = dataset['train'].features['label'].names
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Define image transformations
normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)

# Define function to preprocess images
def preprocess_train(examples):
    examples['pixel_values'] = [train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def preprocess_val(examples):
    examples['pixel_values'] = [val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Apply preprocessing
train_dataset = dataset['train'].with_transform(preprocess_train)
val_dataset = dataset['validation'].with_transform(preprocess_val)
test_dataset = dataset['test'].with_transform(preprocess_val)

# Create data loaders
batch_size = 16  # Smaller batch size for larger images
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
```

### Step 3: Load Pre-trained Model and Modify for Transfer Learning

```python
# Load pre-trained model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=num_classes,
    id2label={str(i): class_names[i] for i in range(num_classes)},
    label2id={class_names[i]: str(i) for i in range(num_classes)}
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Freeze the feature extractor parameters
for param in model.vit.embeddings.parameters():
    param.requires_grad = False

for i in range(8):  # Freeze first 8 layers
    for param in model.vit.encoder.layer[i].parameters():
        param.requires_grad = False
```

### Step 4: Define Training and Evaluation Functions

```python
# Define training function
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        # Get inputs
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader), correct / total

# Define evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Get inputs
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values)

            # Get predictions
            _, predicted = torch.max(outputs.logits, 1)

            # Update statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
```

### Step 5: Train and Evaluate the Model

```python
# Set training parameters
num_epochs = 5
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Lists to store metrics
train_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}:")

    # Train
    train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, scheduler, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Evaluate
    val_accuracy = evaluate(model, val_dataloader, device)
    val_accuracies.append(val_accuracy)

    # Print statistics
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on test set
test_accuracy = evaluate(model, test_dataloader, device)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

### Step 6: Plot Training Progress

```python
# Plot training progress
plt.figure(figsize=(12, 4))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train')
plt.plot(val_accuracies, label='Validation')
plt.axhline(y=test_accuracy, color='r', linestyle='-', label='Test')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

### Step 7: Save the Fine-tuned Model

```python
# Save the model
model.save_pretrained("./vit-flowers")
feature_extractor.save_pretrained("./vit-flowers")
print("Model saved to ./vit-flowers")
```

### Solution Analysis

This exercise demonstrates transfer learning with Vision Transformers on a custom dataset (Flowers-102). Key aspects include:

1. **Parameter Freezing**: By freezing the embedding layer and early transformer blocks, we leverage the pre-trained feature representations while allowing the model to adapt to the new classification task.

2. **Learning Rate Selection**: Using a small learning rate (2e-5) for fine-tuning prevents catastrophic forgetting of the pre-trained knowledge.

3. **Data Augmentation**: Applying random crops and flips to training images helps improve generalization, especially important when working with limited data.

4. **Performance Monitoring**: Tracking both training and validation accuracy helps detect overfitting and determine the optimal number of training epochs.

Transfer learning with ViT is particularly effective for specialized image classification tasks, as the pre-trained model has already learned general visual features that can be adapted to new domains with relatively little training data.

## Exercise 5: Efficient Inference with Vision Transformers

In this exercise, you'll learn how to optimize a Vision Transformer model for efficient inference, which is particularly important for deployment on resource-constrained environments.

### Step 1: Setup Environment

```python
# Install required libraries
!pip install transformers torch torchvision matplotlib optimum onnx onnxruntime

# Import necessary libraries
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import time
import numpy as np
```

### Step 2: Load Pre-trained Model and Test Image

```python
# Load pre-trained ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Load a sample image
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis('off')
plt.show()
```

### Step 3: Benchmark Standard Inference

```python
# Function to measure inference time
def benchmark_inference(model, feature_extractor, image, device, num_runs=10):
    # Prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warm-up run
    with torch.no_grad():
        _ = model(**inputs)

    # Benchmark runs
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            outputs = model(**inputs)
    end_time = time.time()

    # Calculate average time
    avg_time = (end_time - start_time) / num_runs

    # Get prediction
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return avg_time, predicted_class_idx

# Benchmark standard model
std_time, std_pred = benchmark_inference(model, feature_extractor, image, device)
print(f"Standard model inference time: {std_time*1000:.2f} ms")
print(f"Predicted class: {model.config.id2label[std_pred]}")
```

### Step 4: Optimize with Torch JIT

```python
# Create a JIT traced model
def trace_model(model, feature_extractor, device):
    # Prepare dummy input
    dummy_input = feature_extractor(images=Image.new('RGB', (224, 224)), return_tensors="pt")
    dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model, example_kwarg_inputs=dummy_input
        )

    return traced_model

# Trace the model
traced_model = trace_model(model, feature_extractor, device)

# Benchmark JIT traced model
jit_time, jit_pred = benchmark_inference(traced_model, feature_extractor, image, device)
print(f"JIT traced model inference time: {jit_time*1000:.2f} ms")
print(f"Predicted class: {model.config.id2label[jit_pred]}")
print(f"Speed improvement: {std_time/jit_time:.2f}x")
```

### Step 5: Quantize the Model

```python
# Quantize the model to int8
def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

# Try to quantize the model (note: may not work with all models)
try:
    # Move model to CPU for quantization
    cpu_model = model.cpu()

    # Quantize
    quantized_model = quantize_model(cpu_model)

    # Move back to original device
    quantized_model = quantized_model.to(device)

    # Benchmark quantized model
    quant_time, quant_pred = benchmark_inference(quantized_model, feature_extractor, image, device)
    print(f"Quantized model inference time: {quant_time*1000:.2f} ms")
    print(f"Predicted class: {model.config.id2label[quant_pred]}")
    print(f"Speed improvement: {std_time/quant_time:.2f}x")
except Exception as e:
    print(f"Quantization failed: {e}")
    print("Dynamic quantization may not be supported for this model architecture.")
```

### Step 6: Export to ONNX Format

```python
# Export model to ONNX format
def export_to_onnx(model, feature_extractor):
    # Prepare dummy input
    dummy_input = feature_extractor(images=Image.new('RGB', (224, 224)), return_tensors="pt")

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input['pixel_values'],),
        "vit_model.onnx",
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=12
    )

    return "vit_model.onnx"

# Try to export the model to ONNX
try:
    # Move model to CPU for ONNX export
    cpu_model = model.cpu()

    # Export to ONNX
    onnx_path = export_to_onnx(cpu_model, feature_extractor)
    print(f"Model exported to {onnx_path}")

    # Move model back to original device
    model = model.to(device)
except Exception as e:
    print(f"ONNX export failed: {e}")
```

### Step 7: Inference with Batch Processing

```python
# Function to benchmark batch inference
def benchmark_batch_inference(model, feature_extractor, image, device, batch_size=4, num_runs=10):
    # Create a batch of images
    images = [image] * batch_size

    # Prepare batch for the model
    inputs = feature_extractor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warm-up run
    with torch.no_grad():
        _ = model(**inputs)

    # Benchmark runs
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            outputs = model(**inputs)
    end_time = time.time()

    # Calculate average time per image
    avg_time_per_image = (end_time - start_time) / (num_runs * batch_size)

    return avg_time_per_image

# Benchmark batch inference
batch_sizes = [1, 2, 4, 8, 16]
batch_times = []

for bs in batch_sizes:
    try:
        avg_time = benchmark_batch_inference(model, feature_extractor, image, device, batch_size=bs)
        batch_times.append(avg_time)
        print(f"Batch size {bs}: {avg_time*1000:.2f} ms per image")
    except RuntimeError as e:
        print(f"Batch size {bs} failed: {e}")
        break

# Plot batch inference results
plt.figure(figsize=(10, 5))
plt.plot(batch_sizes[:len(batch_times)], [t*1000 for t in batch_times], marker='o')
plt.title('Inference Time per Image vs. Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Time per Image (ms)')
plt.grid(True)
plt.show()
```

### Solution Analysis

This exercise demonstrates various techniques to optimize Vision Transformer models for efficient inference:

1. **JIT Tracing**: Converting the model to TorchScript via tracing can improve inference speed by optimizing the execution graph.

2. **Quantization**: Reducing the precision of model weights from 32-bit floating point to 8-bit integers can significantly decrease memory usage and improve inference speed, with minimal impact on accuracy.

3. **ONNX Export**: Exporting to ONNX format allows the model to be deployed on various hardware and software platforms that support the ONNX runtime.

4. **Batch Processing**: Processing multiple images in a batch can improve throughput by better utilizing hardware parallelism, though there's a trade-off with memory usage.

These optimization techniques are particularly important when deploying Vision Transformers in production environments or on edge devices with limited computational resources. The specific gains will vary depending on the hardware, model size, and implementation details.

## Conclusion

These hands-on exercises provide a comprehensive introduction to working with Vision Transformers for computer vision tasks. From basic inference with pre-trained models to fine-tuning on custom datasets, attention visualization, and optimization for efficient deployment, you've explored the key aspects of using ViTs in practical applications.

Vision Transformers represent a significant advancement in computer vision, offering a different approach from traditional CNNs by leveraging self-attention mechanisms to capture global relationships in images. As demonstrated in these exercises, they can achieve excellent performance across various tasks while providing unique insights through attention visualization.

As you continue working with Vision Transformers, remember that they typically perform best when pre-trained on large datasets and then fine-tuned for specific tasks. The transfer learning approach is particularly effective for adapting these powerful models to specialized domains with limited training data.
