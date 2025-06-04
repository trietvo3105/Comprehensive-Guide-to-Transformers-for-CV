# Transformer Theory for Computer Vision

## Introduction to Transformers

Transformers have revolutionized the field of deep learning since their introduction in the 2017 paper "Attention is All You Need" by Vaswani et al. Originally designed for natural language processing (NLP) tasks, transformers have since been adapted for computer vision applications with remarkable success. This section covers the foundational theory of transformers, with a focus on their application to computer vision tasks.

## The Attention Mechanism: The Core of Transformers

### Self-Attention Explained

The key innovation in transformer models is the self-attention mechanism, which allows the model to weigh the importance of different parts of the input data. Unlike recurrent neural networks (RNNs) or convolutional neural networks (CNNs), which process data sequentially or locally, self-attention can directly model relationships between all positions in a sequence.

In the context of transformers, attention mechanisms serve to weigh the influence of different input tokens when producing an output. The self-attention mechanism computes a weighted sum of all tokens in a sequence, where the weights are determined by the compatibility between the query and key representations of the tokens.

### Scaled Dot-Product Attention

The transformer implements a scaled dot-product attention, which follows these steps:

1. For each position in the input sequence, create three vectors: Query (Q), Key (K), and Value (V)
2. Calculate attention scores by taking the dot product of the query with all keys
3. Scale the scores by dividing by the square root of the dimension of the key vectors
4. Apply a softmax function to obtain the weights
5. Multiply each value vector by its corresponding weight and sum them to produce the output

Mathematically, this is represented as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

Where:
- Q, K, and V are matrices containing the queries, keys, and values
- d_k is the dimension of the key vectors
- The scaling factor √d_k prevents the softmax function from having extremely small gradients

### Multi-Head Attention

To enhance the model's ability to focus on different positions and representation subspaces, transformers use multi-head attention. This involves:

1. Linearly projecting the queries, keys, and values multiple times with different learned projections
2. Performing the attention function in parallel on each projection
3. Concatenating the results and projecting again

This allows the model to jointly attend to information from different representation subspaces at different positions, providing a richer understanding of the input data.

## Transformer Architecture

### Overall Structure

A transformer consists of an encoder and a decoder, each composed of multiple identical layers. Each layer has two main components:

1. A multi-head self-attention mechanism
2. A position-wise fully connected feed-forward network

Additionally, each sublayer employs residual connections and layer normalization to facilitate training.

### The Encoder

The encoder processes the input sequence and generates representations that capture the contextual relationships within the data. Each encoder layer consists of:

1. Multi-head self-attention layer: Allows each position to attend to all positions in the previous layer
2. Feed-forward neural network: Applied to each position separately and identically
3. Layer normalization and residual connections: Stabilize and accelerate training

### The Decoder

The decoder generates the output sequence one element at a time. Each decoder layer includes:

1. Masked multi-head self-attention: Prevents positions from attending to subsequent positions
2. Multi-head attention over the encoder output: Allows the decoder to focus on relevant parts of the input sequence
3. Feed-forward neural network
4. Layer normalization and residual connections

### Positional Encoding

Since transformers do not inherently process sequential information, positional encodings are added to the input embeddings to provide information about the relative or absolute position of tokens in the sequence. These encodings have the same dimension as the embeddings, allowing them to be summed.

## From NLP to Computer Vision: Vision Transformers (ViT)

### Adapting Transformers for Images

Vision Transformers (ViT) adapt the transformer architecture for image processing tasks. The key insight is to treat an image as a sequence of patches, similar to how words are treated in NLP applications.

The process involves:

1. Splitting the image into fixed-size patches (typically 16×16 pixels)
2. Flattening each patch into a vector
3. Linearly projecting these vectors to obtain patch embeddings
4. Adding positional embeddings to retain spatial information
5. Processing the resulting sequence with a standard transformer encoder

### ViT Architecture Details

The Vision Transformer architecture includes:

1. **Patch Embedding**: Images are divided into patches, which are linearly embedded
2. **Class Token**: A learnable embedding is prepended to the sequence of embedded patches (similar to BERT's [CLS] token)
3. **Positional Embedding**: Added to the patch embeddings to retain spatial information
4. **Transformer Encoder**: Standard transformer encoder blocks process the sequence
5. **MLP Head**: A classification head is attached to the output of the [CLS] token for image classification tasks

### Key Differences from CNNs

Vision Transformers differ from traditional CNNs in several important ways:

1. **Global Processing**: ViTs process the entire image globally from the first layer, whereas CNNs build global context gradually through hierarchical processing
2. **Inductive Bias**: CNNs have strong inductive biases (locality, translation equivariance), while ViTs have minimal image-specific inductive biases
3. **Positional Information**: CNNs implicitly encode spatial relationships, while ViTs must learn these relationships through positional embeddings
4. **Computational Efficiency**: ViTs can be more computationally efficient for certain tasks, especially when pre-trained on large datasets

## Advanced Transformer Variants for Computer Vision

### Swin Transformer

The Swin (Shifted Window) Transformer introduces hierarchical feature maps and shifted windowing to improve efficiency and performance. Key innovations include:

1. **Hierarchical Feature Maps**: Merging patches progressively to create a hierarchical representation
2. **Window-based Self-Attention**: Computing self-attention within local windows to reduce computational complexity
3. **Shifted Window Partitioning**: Alternating between regular and shifted window partitioning to enable cross-window connections

### DeiT (Data-efficient image Transformers)

DeiT demonstrates that Vision Transformers can be trained effectively on smaller datasets through:

1. **Distillation Token**: A specific token that learns from a teacher model (typically a CNN)
2. **Strong Data Augmentation**: Techniques like RandAugment and CutMix to increase data diversity
3. **Regularization**: Methods to prevent overfitting on smaller datasets

### MobileViT

MobileViT combines the strengths of CNNs and transformers for mobile applications:

1. **Local Processing**: Using convolutions for local feature extraction
2. **Global Understanding**: Applying transformers for global context modeling
3. **Lightweight Design**: Optimized architecture for mobile devices with limited computational resources

## Attention Mechanisms Beyond Transformers

### Non-local Neural Networks

Non-local neural networks capture long-range dependencies by computing the response at a position as a weighted sum of features at all positions. This is conceptually similar to self-attention but was developed independently.

### Squeeze-and-Excitation Networks

Squeeze-and-Excitation networks adaptively recalibrate channel-wise feature responses by explicitly modeling interdependencies between channels, providing a form of attention over feature channels.

### CBAM (Convolutional Block Attention Module)

CBAM combines channel and spatial attention to enhance the representational power of CNNs, focusing on "what" and "where" to attend in the feature maps.

## Conclusion

Transformers have fundamentally changed how we approach computer vision tasks. By leveraging self-attention mechanisms, they can capture global dependencies and relationships within images more effectively than traditional architectures. Vision Transformers and their variants have demonstrated state-of-the-art performance across various computer vision tasks, from image classification to object detection and segmentation.

The success of transformers in computer vision highlights the value of cross-pollination between different domains of deep learning. As research continues, we can expect further innovations that combine the strengths of transformers with other architectural paradigms, leading to more efficient and effective models for computer vision applications.

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems.

2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations.

3. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

4. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). Training data-efficient image transformers & distillation through attention. In International Conference on Machine Learning.

5. Mehta, S., & Rastegari, M. (2021). MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer. arXiv preprint arXiv:2110.02178.

6. Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition.

7. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition.

8. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). Cbam: Convolutional block attention module. In Proceedings of the European conference on computer vision.

9. Alammar, J. (2018). The Illustrated Transformer. https://jalammar.github.io/illustrated-transformer/

10. Machine Learning Mastery. (2023). The Transformer Attention Mechanism. https://www.machinelearningmastery.com/the-transformer-attention-mechanism/
