# Styling Guide for Transformer Lecture Website

This document provides instructions on how to use the various styling features available in your Transformer lecture website.

## Table of Contents

1. [Images with Captions](#images-with-captions)
2. [Image Gallery](#image-gallery)
3. [Callout Boxes](#callout-boxes)
4. [Card Layout](#card-layout)
5. [Math Equations](#math-equations)
6. [Code Blocks](#code-blocks)
7. [Tables](#tables)

## Images with Captions

To add an image with a caption, use the `figure.html` include:

{% raw %}

```liquid
{% include figure.html
   src="path/to/your/image.jpg"
   alt="Alternative text"
   caption="Your caption here"
   width="600px"  <!-- Optional -->
%}
```

{% endraw %}

## Image Gallery

To create an image gallery, use the following code:

{% raw %}

```liquid
{% assign gallery_images = "" | split: "" %}
{% assign image1 = "" | split: "" | push: "path/to/image1.jpg" | push: "Alt text 1" | push: "Caption 1" %}
{% assign image2 = "" | split: "" | push: "path/to/image2.jpg" | push: "Alt text 2" | push: "Caption 2" %}
{% assign image3 = "" | split: "" | push: "path/to/image3.jpg" | push: "Alt text 3" | push: "Caption 3" %}

{% assign gallery_images = gallery_images | push: image1 | push: image2 | push: image3 %}

<div class="image-gallery">
  {% for image in gallery_images %}
    <div class="gallery-item">
      <img src="{{ image[0] }}" alt="{{ image[1] }}" />
      <span class="image-caption">{{ image[2] }}</span>
    </div>
  {% endfor %}
</div>
```

{% endraw %}

## Callout Boxes

To create a callout box, use the `callout.html` include:

{% raw %}

```liquid
{% capture callout_content %}
Your content here. This can include **markdown** formatting.
{% endcapture %}
{% include callout.html content=callout_content title="Optional Title" %}
```

{% endraw %}

You can also create different types of callout boxes:

{% raw %}

```liquid
{% include callout.html content=warning_content title="Warning" type="warning" %}
{% include callout.html content=success_content title="Success" type="success" %}
{% include callout.html content=danger_content title="Danger" type="danger" %}
```

{% endraw %}

## Card Layout

To create a card layout, use the `card.html` include:

{% raw %}

```liquid
{% capture card_content %}
Your content here. This can include **markdown** formatting.
{% endcapture %}
{% include card.html title="Card Title" content=card_content %}
```

{% endraw %}

## Math Equations

You can include math equations using LaTeX syntax:

```markdown
When $a \ne 0$, there are two solutions to $ax^2 + bx + c = 0$ and they are
$$x = {-b \pm \sqrt{b^2-4ac} \over 2a}.$$
```

## Code Blocks

For code blocks, use triple backticks with the language name:

````markdown
```python
# This is a Python code block
def hello_world():
    print("Hello, Vision Transformer world!")
```
````

````

## Tables

Create tables using the standard Markdown syntax:

```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |
````

## See It In Action

Check out the [Styling Demo](styling-demo.md) page to see all these features in action.
