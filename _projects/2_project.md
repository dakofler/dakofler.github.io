---
layout: page
title: Vision Transformer
description: Implementation of the Vision Transformer Architecture in PyTorch
img: assets/img/vit.jpg
importance: 2
category: Uni
---

# Vision Transformer

An Implemenentation of the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929) in Pytorch.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/vit.jpg" title="Vision Transformer by Dosovitskiy et al." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Vision Transformer by Dosovitskiy et al.
</div>

## Installation

- Clone the repository
```bash
git clone https://github.com/dakofler/vision_transformer.git
cd vision_transformer/
```
- (Optional) Create a Python virtual environment (Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
```
- Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
- Download the [CIFAR10-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset
- Upack and put the files into a `./data` directory

```bash
vision_transformer/
├─ data/
│  ├─ batches.meta
│  ├─ data_batch_1
│  ├─ data_batch_2
│  ├─ data_batch_3
│  ├─ data_batch_4
│  ├─ data_batch_5
│  ├─ readme.html
│  ├─ test_batch
...
```
- run the training script

```bash
python3 train.py
```