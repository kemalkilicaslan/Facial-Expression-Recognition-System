# Facial Expression Recognition System

## 1. Introduction

This project implements a deep learning-based facial expression recognition system using the FER-2013 dataset and VGG19 architecture with batch normalization. The system can classify facial expressions into seven distinct emotion categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

Emotion recognition is used across domains such as human-computer interaction, mental health monitoring, customer sentiment analysis, and entertainment. Using transfer learning with VGG19 and custom classifier heads, the system works on grayscale facial images.

It uses data augmentation to improve generalization and reduce overfitting on the small 48×48 pixel images.

**Core Features:**
- 7-class emotion classification (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- Transfer learning with pre-trained VGG19 architecture
- Data augmentation to improve generalization
- Batch normalization for training stability
- Dropout regularization to prevent overfitting
- Training visualization and evaluation metrics
- Individual prediction visualization with probability distributions

## 2. Methodology / Approach

The system employs transfer learning with VGG19, a deep convolutional neural network pre-trained on ImageNet. The architecture is adapted for facial expression recognition by replacing the final classification layers with custom fully-connected layers tailored for 7 emotion classes.

### 2.1 System Architecture

The facial expression recognition pipeline consists of:

1. **Data Loading:** FER-2013 dataset with 35,887 grayscale images (48×48 pixels)
2. **Data Augmentation:** Random horizontal flips and rotations for training set
3. **Model Architecture:** VGG19 backbone with custom classifier head
4. **Training:** AdamW optimizer with cross-entropy loss
5. **Evaluation:** Accuracy tracking and loss monitoring across epochs
6. **Prediction:** Softmax probabilities for emotion classification

### 2.2 Implementation Strategy

The implementation uses PyTorch as the deep learning framework with the `timm` library for accessing pre-trained models. Data augmentation techniques (horizontal flips, rotations) are applied during training to improve model robustness. The custom classifier includes dropout layers (30% rate) to prevent overfitting, and gradient clipping is employed to stabilize training.

## 3. Mathematical Framework

### 3.1 Model Architecture

The VGG19 architecture consists of:
- **Feature Extractor:** 16 convolutional layers + 3 fully connected layers
- **Custom Classifier:**

$$\text{Input} \xrightarrow{\text{Dropout}(0.3)} \text{Linear}(4096 \to 512) \xrightarrow{\text{ReLU}} \text{Dropout}(0.3) \xrightarrow{} \text{Linear}(512 \to 7)$$

### 3.2 Loss Function

Cross-entropy loss for multi-class classification:

$$\mathcal{L} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \cdot \log(p_{ic})$$

where:
- $N$ = batch size
- $C = 7$ emotion classes
- $y_{ic}$ = ground truth (1 if sample $i$ belongs to class $c$, 0 otherwise)
- $p_{ic}$ = predicted probability for sample $i$ and class $c$

### 3.3 Accuracy Calculation

$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\arg\max(p_i) = y_i]$$

where $\mathbb{1}$ is the indicator function returning 1 for correct predictions.

### 3.4 Data Augmentation

Training transformations:
- Random horizontal flip: probability $p = 0.5$
- Random rotation: range $\theta \in [-20°, +20°]$
- Tensor normalization: $x' = \frac{x}{255}, \quad x \in [0, 255] \Rightarrow x' \in [0, 1]$

## 4. Requirements

`requirements.txt`

```txt
numpy>=1.19.0
matplotlib>=3.3.0
plotly>=5.0.0
torch>=1.9.0
torchvision>=0.10.0
timm>=0.4.12
tqdm>=4.62.0
pillow>=8.0.0
```

## 5. Installation & Configuration

### 5.1 Environment Setup

```bash
# Clone the repository
git clone https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System.git
cd Facial-Expression-Recognition-System

# Install required packages
pip install -r requirements.txt
```

### 5.2 Project Structure

```
Facial-Expression-Recognition-System
├── test/                          # Test dataset (7,178 images)
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── train/                         # Training dataset (28,709 images)
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── Facial-Expression-Recognition-System.ipynb
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

### 5.3 Dataset Information

**FER-2013 Dataset:**
- Total images: 35,887 (48×48 pixels, grayscale)
- Training set: 28,709 images (80%)
- Test set: 7,178 images (20%)
- Classes: 7 emotions
  - 0: Angry
  - 1: Disgust
  - 2: Fear
  - 3: Happy
  - 4: Neutral
  - 5: Sad
  - 6: Surprise

## 6. Usage / How to Run

### 6.1 Training the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook Facial-Expression-Recognition-System.ipynb
```

Or run cells sequentially in Google Colab after uploading the notebook.

### 6.2 Configuration Parameters

```python
# Hyperparameters (can be modified in the notebook)
lr = 0.0001              # Learning rate
batch_size = 16          # Batch size for training
epochs = 20              # Number of training epochs
device = 'cuda'          # 'cuda' for GPU, 'cpu' for CPU
model_name = 'vgg19_bn'  # Model architecture
dropout_rate = 0.3       # Dropout probability
```

### 6.3 Model Training Process

```python
# Initialize model
model = FaceRecognitionModel(dropout_rate=0.3)
model.to(device)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# Train the model
for epoch in range(epochs):
    train_loss, train_acc = train_func(model, trainloader, optimizer, epoch)
    test_loss, test_acc = eval_func(model, testloader, epoch)
    
    # Save best model
    if test_loss < best_test_loss:
        torch.save(model.state_dict(), 'best-weights.pt')
```

### 6.4 Making Predictions

```python
# Load best model
model.load_state_dict(torch.load('best-weights.pt'))

# Make predictions
predict(model, testloader, num_class=7)
```

## 7. Application / Results

### 7.1 Sample Training Images

**Training Set Example:**

![Training Sample](https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System/blob/main/trainset40.png)

**Test Set Example:**

![Test Sample](https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System/blob/main/testset40.png)

### 7.2 Training History

**Loss and Accuracy Over 20 Epochs:**

![Training History](https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System/blob/main/training-history-plot.png)

### 7.3 Prediction Results

**Angry Expression:**

![angry](https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System/blob/main/angry.png)

- True Label: Angry
- Predicted Label: Angry
- Confidence: 59.37%

**Disgust Expression:**

![disgust](https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System/blob/main/disgust.png)

- True Label: Disgust
- Predicted Label: Surprise
- Confidence: 60.77%

**Fear Expression:**

![fear](https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System/blob/main/fear.png)

- True Label: Fear
- Predicted Label: Fear
- Confidence: 37.67%

**Happy Expression:**

![happy](https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System/blob/main/happy.png)

- True Label: Happy
- Predicted Label: Happy
- Confidence: 98.41%

**Neutral Expression:**

![neutral](https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System/blob/main/neutral.png)

- True Label: Neutral
- Predicted Label: Neutral
- Confidence: 50.31%

**Sad Expression:**

![sad](https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System/blob/main/sad.png)

- True Label: Sad
- Predicted Label: Fear
- Confidence: 40.81%

**Surprise Expression:**

![surprise](https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System/blob/main/surprise.png)

- True Label: Surprise
- Predicted Label: Surprise
- Confidence: 96.73%

### 7.4 Performance Metrics

**Final Model Performance (Epoch 10 - Best Model):**

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| Loss | 0.8215 | 0.9448 |
| Accuracy | 70.51% | 66.33% |

**Per-Emotion Performance:**

| Emotion | Recognition Quality |
|---------|-------------------|
| Happy | Excellent (98.41%) |
| Surprise | Excellent (96.73%) |
| Angry | Good (59.37%) |
| Neutral | Moderate (50.31%) |
| Fear | Moderate (37.67%) |
| Sad | Challenging (misclassified as Fear) |
| Disgust | Challenging (misclassified as Surprise) |

## 8. Tech Stack

### 8.1 Core Technologies

- **Programming Language:** Python 3.7+
- **Deep Learning Framework:** PyTorch 1.9+
- **Model Library:** timm (PyTorch Image Models)
- **Dataset:** FER-2013 (Facial Expression Recognition 2013)

### 8.2 Libraries & Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| torch | 1.9+ | Deep learning framework |
| torchvision | 0.10+ | Image transformations and datasets |
| timm | 0.4.12+ | Pre-trained model access (VGG19) |
| numpy | 1.19+ | Numerical computations |
| matplotlib | 3.3+ | Static visualization |
| plotly | 5.0+ | Interactive visualization |
| tqdm | 4.62+ | Progress bar for training |
| pillow | 8.0+ | Image processing |

### 8.3 Model Architecture

**VGG19 with Batch Normalization:**
- **Backbone:** VGG19_bn (pre-trained on ImageNet)
- **Feature Layers:** 53 layers total
  - Convolutional layers: 16
  - Batch normalization: Multiple
  - Max pooling: 5
  - Fully connected: 3 (pre-logits)
- **Custom Head:** 2-layer MLP with dropout
- **Total Parameters:** ~144M (backbone) + custom classifier
- **Output:** 7-class softmax probabilities

### 8.4 Training Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Optimizer | AdamW | Improved Adam with weight decay |
| Learning Rate | 0.0001 | Step size for gradient descent |
| Weight Decay | 0.01 | L2 regularization |
| Batch Size | 16 | Mini-batch size |
| Epochs | 20 | Training iterations |
| Dropout Rate | 0.3 | Prevent overfitting |
| Gradient Clipping | 1.0 | Prevent gradient explosion |

## 9. License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://github.com/kemalkilicaslan/Facial-Expression-Recognition-System/blob/main/LICENSE).

## 10. References

1. Kaggle [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013).
2. PyTorch [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) Documentation.
3. Hugging Face [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models/tree/main/timm) GitHub Repository.

---

### Acknowledgments

This project uses the FER-2013 dataset created for the Facial Expression Recognition Challenge. Thanks to the PyTorch and timm communities for their deep learning tools and pre-trained models. The VGG19 architecture was developed by the Visual Geometry Group at the University of Oxford.

---

**Note:** This system is designed for research and educational purposes. Facial expression recognition should be used responsibly and ethically, with consideration for privacy, consent, and potential biases in emotion detection. The model's performance varies across different emotions, with some expressions (like "disgust" and "sad") being more challenging to classify accurately.