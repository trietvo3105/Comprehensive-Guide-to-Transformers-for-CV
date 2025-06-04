---
layout: default
title: Styling Demo
---

# Styling Demo

This page demonstrates the various styling options available for your Transformer lecture website.

## Images with Captions

Here's how to use the figure include to add images with captions:

{% include figure.html
   src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Example_image.svg/600px-Example_image.svg.png"
   alt="Example image"
   caption="This is a caption for the example image"
%}

## Image Gallery

You can create an image gallery like this:

{% assign gallery_images = "" | split: "" %}
{% assign image1 = "" | split: "" | push: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Example_image.svg/600px-Example_image.svg.png" | push: "Example image 1" | push: "Caption for image 1" %}
{% assign image2 = "" | split: "" | push: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Example_image.svg/600px-Example_image.svg.png" | push: "Example image 2" | push: "Caption for image 2" %}
{% assign image3 = "" | split: "" | push: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Example_image.svg/600px-Example_image.svg.png" | push: "Example image 3" | push: "Caption for image 3" %}

{% assign gallery_images = gallery_images | push: image1 | push: image2 | push: image3 %}

<div class="image-gallery">
  {% for image in gallery_images %}
    <div class="gallery-item">
      <img src="{{ image[0] }}" alt="{{ image[1] }}" />
      <span class="image-caption">{{ image[2] }}</span>
    </div>
  {% endfor %}
</div>

## Callout Boxes

You can create callout boxes for important information:

{% capture callout_content %}
This is a standard callout box that can be used to highlight important information.
{% endcapture %}
{% include callout.html content=callout_content title="Important Note" %}

{% capture warning_content %}
This is a warning callout that can be used to highlight potential issues or warnings.
{% endcapture %}
{% include callout.html content=warning_content title="Warning" type="warning" %}

{% capture success_content %}
This is a success callout that can be used to highlight achievements or successful outcomes.
{% endcapture %}
{% include callout.html content=success_content title="Success" type="success" %}

## Card Layout

You can use cards to organize content into distinct sections:

{% capture card_content %}
This is the content of the card. You can include any markdown content here, such as:

-   Lists
-   **Bold text**
-   _Italic text_
-   [Links](https://example.com)

And even code blocks:

```python
def hello_world():
    print("Hello, Vision Transformer world!")
```

{% endcapture %}
{% include card.html title="Example Card" content=card_content %}

## Math Equations

You can include math equations using LaTeX syntax:

When $a \ne 0$, there are two solutions to $ax^2 + bx + c = 0$ and they are
$$x = {-b \pm \sqrt{b^2-4ac} \over 2a}.$$

## Code Blocks

```python
# This is a Python code block
def attention_mechanism(query, key, value):
    """
    Implement a simple attention mechanism
    """
    scores = query.dot(key.transpose())
    attention_weights = softmax(scores)
    output = attention_weights.dot(value)
    return output, attention_weights
```

## Tables

| Model     | Parameters | Top-1 Accuracy | Training Time |
| --------- | ---------- | -------------- | ------------- |
| ViT-Base  | 86M        | 84.2%          | 2.5 days      |
| ViT-Large | 307M       | 85.3%          | 4 days        |
| Swin-T    | 29M        | 81.3%          | 1.5 days      |
| Swin-S    | 50M        | 83.0%          | 2 days        |

## Responsive Design

The website is fully responsive and will adapt to different screen sizes. Try resizing your browser window to see how the layout changes.
