# EchoScript

EchoScript is an advanced audio captioning system that leverages deep learning to generate descriptive natural language captions for audio clips. 

It utilizes a hybrid architecture featuring an HTSAT encoder and a BART decoder to process complex audio environments and describe them through an immersive web interface.

---

## Features

### Deep Learning Powered
Uses a pre-trained HTSAT (Hierarchical Token-Semantic Audio Transformer) for robust audio feature extraction.

### Natural Language Generation
Employs a BART decoder to translate audio features into coherent, human-like captions.

### Real-Time Web Interface
Flask-based application featuring a cinematic landing page with 3D Spline integration.

### Beam Search Reasoning
Generates the top 3 candidate captions with probability scores to expose the model's decision-making process.

### Automated Audio Handling
Automatically manages:
- Resampling
- Mono conversion
- 10-second fixed-length padding or truncation

---

## Architecture Overview

- **Encoder**: HTSAT (Hierarchical Token-Semantic Audio Transformer)
- **Decoder**: BART
- **Framework**: PyTorch
- **Inference Strategy**: Beam Search

---

## Tech Stack

### Backend
- Flask (Python)

### Deep Learning
- PyTorch
- Hugging Face Transformers

### Audio Processing
- Librosa

### Frontend
- HTML5
- CSS3
- JavaScript
- Spline

---

## Project Structure
EchoScript/
│
├── app.py
├── notebooks/
├── models/ # Model weights (not included)
├── static/
├── templates/
├── requirements.txt
└── README.md

---

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/PBAniket/EchoScript.git
cd EchoScript

pip install torch librosa flask transformers werkzeug
