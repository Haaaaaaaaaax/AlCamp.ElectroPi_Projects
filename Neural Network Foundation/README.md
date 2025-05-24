# Neural Network Foundation - Price Prediction Model

This project implements a machine learning model to predict product prices based on textual descriptions, quantities, and country information. It leverages BERT embeddings for text representation and neural networks for regression.

## Project Overview

The model aims to predict the `UnitPrice` of products using the following features:
- Product Description (text)
- Quantity
- Country

## Technical Implementation

### Data Processing
- Dataset size: ~500,000 samples
- Text descriptions are limited to 10 words per sentence
- Handles missing values and outliers
- Removes duplicates for better model performance

### Model Architecture
1. **Text Embedding**
   - Uses Hugging Face's BERT model (`bert-base-uncased`)
   - Tokenizes text with max length of 50
   - Generates embeddings through mean pooling
   - GPU-accelerated batch processing

2. **Dimensionality Reduction**
   - Applies PCA to reduce BERT embeddings to 20 dimensions
   - Uses StandardScaler for feature normalization

3. **Neural Network**
   - Feedforward architecture with 2 hidden layers (128, 64 neurons)
   - ReLU activation functions
   - Dropout (0.3) for regularization
   - Single output neuron for regression
   - Optimizer: Adamax (learning rate: 1e-4)
   - Loss function: Mean Squared Error

### Model Performance
- RMSE: 0.6485
- MAE: 0.4688
- R² Score: 0.5806

## Requirements

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- numpy
- pandas
- tqdm


## Future Improvements

- Increase PCA dimensions for better feature representation
- Implement regularization techniques
- Use hyperparameter tuning (e.g., Optuna)
- Explore more advanced architectures
- Consider using different pre-trained models

## Project Structure

- `NoteBooks/`
  - `EDA.ipynb`: Exploratory Data Analysis
  - `Preprocess.ipynb`: Data preprocessing pipeline
  - `train.ipynb`: Model training and evaluation
- `Model docs.txt`: Detailed model documentation
- `ANN.pth`: Saved model weights
