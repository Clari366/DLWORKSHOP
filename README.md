# DLWORKSHOP

# Income Classification using PyTorch

This project demonstrates how to build a **deep learning model with embeddings for categorical data** to predict whether a person’s income is greater than or less than 50K.  

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

## ⚙️ Steps in the Project
1. Load and shuffle the dataset using **pandas**.  
2. Split features into:
   - Categorical columns  
   - Continuous columns  
   - Target label  
3. Convert categorical variables into **embeddings**.  
4. Define a custom **TabularModel** using PyTorch.  
5. Train the model with **CrossEntropyLoss** and **Adam optimizer**.  
6. Evaluate model performance on a test set.  
7. Add a function to make predictions on new input data.  

## 📊 Results
- Training completed over **300 epochs**.  
- Final accuracy: **~84.6%** on the test set.  

## 🖼️ Training Loss
The loss decreases steadily during training:  
