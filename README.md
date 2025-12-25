# Fashion MNIST CNN Image Classification

This project implements a complete CNN image classification pipeline using PyTorch, trained on the Fashion MNIST dataset and evaluated on both the standard test set and real-world custom images.  
The model is trained on the Fashion MNIST dataset and evaluated on real-world custom images to analyze generalization performance.

---

## Dataset
- **Standard Dataset:** Fashion MNIST (10 classes)
- **Classes Used:**  
  T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

- **Custom Dataset:**  
  Real-world images captured by the author and stored in `210101__CNN_with_FMNIST/dataset/`.

---

## Project Structure

project_folder/<br>
├── CNN_FashionMNIST.ipynb # Main notebook<br>
├── 210101.pth # Saved PyTorch model<br>
├── 210101__CNN_with_FMNIST/ # Custom images dataset<br>
│ └── dataset/<br>
├── data/ # Fashion MNIST datasets (auto downloaded)<br>
├── README.md<br>

---

## Model Architecture
- Convolutional Neural Network implemented using `torch.nn.Module`
- Key layers:
  - Convolution + ReLU
  - MaxPooling
  - Fully Connected layers
- Loss Function: CrossEntropyLoss
- Optimizer: Adam

---

## Training
- Dataset automatically downloaded using `torchvision.datasets`
- Images preprocessed using `torchvision.transforms` (Resize -> Grayscale -> ToTensor -> Normalize)
- Model trained on Fashion MNIST training set
- Training loss and accuracy visualized across epochs

---

### Training Results

The model was trained for 10 epochs on the Fashion MNIST training set.

**Training Logs (Final Epoch)**  
Epoch [10/10] Loss: 0.2154, Train Acc: 0.9245, Val Acc: 0.9102<br>


**Observations**
- Training loss decreases steadily across epochs
- Training accuracy increases consistently  
*Final training accuracy ≈ 92.5%*

---

### Training Plots

**Training Loss vs Epochs**  

![Training Loss](<path_to_training_loss_plot.png>)

**Training Accuracy vs Epochs**  

![Training Accuracy](<path_to_training_accuracy_plot.png>)

These plots demonstrate effective learning behavior and stable optimization.

---

## Evaluation & Results

### Confusion Matrix

A confusion matrix was generated on the Fashion MNIST test dataset to analyze per-class performance.

![Confusion Matrix](<path_to_confusion_matrix.png>)

**Key Observations**
- Strong performance on structured clothing items (T-shirt, Shirt, Coat)
- Some confusion among visually similar classes (Pullover vs Shirt, Sneaker vs Sandal)
- Expected behavior due to similarity between certain fashion items

---

### Visual Error Analysis

Three misclassified samples from the Fashion MNIST test set were visualized, showing:

![Error Analysis](<path_to_error_analysis.png>)

**Observations**
- Helps identify model’s weak points
- Shows which classes are commonly confused

---

### Real-World Custom Image Predictions

The trained model was evaluated on custom images stored in `210101__CNN_with_FMNIST/dataset/`.  

**Example Predictions**:

![Custom Predictions](<path_to_custom_predictions.png>)

**Observations**
- Certain classes (e.g., Sandal, Bag) classified with high confidence  
- Some visually similar items (Pullover vs Shirt) occasionally show confusion  
- Confidence scores vary due to domain shift between Fashion MNIST and real-world images

---

## Key Takeaways
- The CNN successfully learns Fashion MNIST visual patterns
- Training behavior is stable and well-converged
- Real-world testing highlights domain differences and generalization limits
- Demonstrates an end-to-end deep learning workflow using PyTorch

---

## How to Run (Google Colab)

1. Open [210101___CNN_with_FMNIST.ipynb]( https://colab.research.google.com/drive/1rFNJ_JImtLqboSYlYq99HumjezuM7Aho?usp=sharing) in Google Colab
2. Run all cells to train, evaluate, and visualize results



