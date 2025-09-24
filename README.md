# DLWORKSHOP

# Income Classification using PyTorch
🎯 Aim

To build a deep learning model using PyTorch with embeddings for categorical features to classify whether a person’s income is greater than 50K or less than or equal to $50K based on census data.

## 📂 Dataset
- The dataset used is `income.csv` (based on the UCI Adult dataset).
- It contains demographic and work-related attributes such as:
  - Age  
  - Sex  
  - Education  
  - Marital Status  
  - Workclass  
  - Occupation  
  - Hours per week  
  - Income label (`<=50K` or `>50K`)  

⚙️ Procedure
1. Dataset Preparation

Used the UCI Adult Census Income dataset.

Features include demographic and work-related details:

Age, Sex, Education, Marital Status, Workclass, Occupation, Hours-per-week

Target: Income (<=50K or >50K)

Data was shuffled, categorical columns converted to categories, and continuous features normalized.

2. Feature Engineering

Categorical Columns:

Converted into embeddings using nn.Embedding.

Continuous Columns:

Batch normalized with nn.BatchNorm1d.

3. Model Architecture

Custom TabularModel built using PyTorch.

Layers:

Embedding layers for categorical features

BatchNorm + Linear layers for continuous features

Fully connected hidden layer(s) with ReLU and Dropout

Output layer with 2 classes (<=50K, >50K)

4. Training

Loss Function: CrossEntropyLoss

Optimizer: Adam

Epochs: 300

Batch size: Full dataset in memory (tabular data is small).

5. Evaluation

Accuracy measured on a test set of 5000 samples.

Prediction function predict_new() added to test new input samples.

📊 Results
Training Loss

Initial Loss (Epoch 1): ~0.81

Final Loss (Epoch 300): ~0.31

Test Performance

Cross Entropy Loss: ~0.33

Accuracy: 84.6% (4232 out of 5000 correct) 
