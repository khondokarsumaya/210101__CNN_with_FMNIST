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
### Training Log (Per Epoch)
Epoch [1/10], Train Loss: 0.6171, Val Loss: 0.3951, Train Acc: 0.8611, Val Acc: 0.8536<br>
Epoch [2/10], Train Loss: 0.4116, Val Loss: 0.3460, Train Acc: 0.8853, Val Acc: 0.8722<br>
Epoch [3/10], Train Loss: 0.3562, Val Loss: 0.3240, Train Acc: 0.8934, Val Acc: 0.8784<br>
Epoch [4/10], Train Loss: 0.3247, Val Loss: 0.3044, Train Acc: 0.9013, Val Acc: 0.8876<br>
Epoch [5/10], Train Loss: 0.3008, Val Loss: 0.2791, Train Acc: 0.9154, Val Acc: 0.8988<br>
Epoch [6/10], Train Loss: 0.2816, Val Loss: 0.2708, Train Acc: 0.9181, Val Acc: 0.9027<br>
Epoch [7/10], Train Loss: 0.2648, Val Loss: 0.2596, Train Acc: 0.9257, Val Acc: 0.9073<br>
Epoch [8/10], Train Loss: 0.2508, Val Loss: 0.2637, Train Acc: 0.9258, Val Acc: 0.9042<br>
Epoch [9/10], Train Loss: 0.2383, Val Loss: 0.2594, Train Acc: 0.9317, Val Acc: 0.9055<br>
Epoch [10/10], Train Loss: 0.2280, Val Loss: 0.2458, Train Acc: 0.9388, Val Acc: 0.9112<br>

**Training Logs (Final Epoch)**  
Epoch [10/10], Train Loss: 0.2280, Val Loss: 0.2458, Train Acc: 0.9388, Val Acc: 0.911


**Observations**
- Training loss decreases steadily across epochs
- Training accuracy increases consistently  
*Final training accuracy ≈ 92.5%*

---

### Training Plots

**Training Loss vs Epochs**  
<img width="855" height="393" alt="download" src="https://github.com/user-attachments/assets/beec54ee-d567-4e48-9a93-75762eabd7d0" />


**Training Accuracy vs Epochs**  

<img width="855" height="393" alt="download" src="https://github.com/user-attachments/assets/3b8b9955-18f2-486c-a03d-518de5e8acdd" />


These plots demonstrate effective learning behavior and stable optimization.

---

## Evaluation & Results

### Confusion Matrix

A confusion matrix was generated on the Fashion MNIST test dataset to analyze per-class performance.

<img width="788" height="701" alt="download" src="https://github.com/user-attachments/assets/1adc7c94-d306-4ba7-a917-069c91d57005" />

**Key Observations**
- Strong performance on structured clothing items (T-shirt, Shirt, Coat)
- Some confusion among visually similar classes (Pullover vs Shirt, Sneaker vs Sandal)
- Expected behavior due to similarity between certain fashion items

---

### Visual Error Analysis

Three misclassified samples from the Fashion MNIST test set were visualized, showing:

<img width="990" height="376" alt="download" src="https://github.com/user-attachments/assets/d9ade8eb-bc16-430d-99f4-56fe310a0d9d" />


**Observations**
- Helps identify model’s weak points
- Shows which classes are commonly confused

---

### Real-World Custom Image Predictions

The trained model was evaluated on custom images stored in `210101__CNN_with_FMNIST/dataset/`.  

**Example Predictions**:
<img width="1189" height="608" alt="download" src="https://github.com/user-attachments/assets/d5549aad-827d-4be8-90fd-7a1bd1e25a01" />



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



