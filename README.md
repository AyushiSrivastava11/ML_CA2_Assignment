# Spam Classification Project

## Overview
This project focuses on building a machine learning model to classify text messages as either "spam" or "not spam" (ham). The dataset used for this task contains labeled messages, which will help us train and evaluate the model.

## Dataset
The dataset used for this project was downloaded from [Kaggle](https://www.kaggle.com/). It contains the following key features:
- `label`: The class label, either spam or ham.
- `message`: The text content of the message.

## Exploratory Data Analysis (EDA)
Before building the model, we performed exploratory data analysis to understand the dataset better. Key steps involved:
- Checking for missing data.
- Visualizing the distribution of spam vs ham messages.
- Tokenizing the messages and creating word clouds to analyze common words in each category.

## Machine Learning Model
We used a variety of machine learning models to approach the problem, including:
- **Naive Bayes Classifier**: A commonly used algorithm for text classification problems.
- **Support Vector Machines (SVM)**: To explore alternative methods for classification.
- **Logistic Regression**: A simple yet effective model for binary classification tasks.

We also used `TF-IDF` to convert the text into numerical features before feeding them into the models.

### Model Evaluation
The models were evaluated using the following metrics:
- **Accuracy**: The percentage of correctly classified messages.
- **Precision**: How many of the messages we labeled as spam were actually spam.
- **Recall**: How many spam messages were correctly identified.
- **F1-Score**: A balance between precision and recall.

The final model provided an accuracy score of **XX%**, with good performance in both precision and recall.

## How to Run the Project
   ```bash
   git clone https://github.com/yourusername/spam-classification
   cd spam-classification
   jupyter notebook Spam_Classification.ipynb
```
### MKey Libraries Used
- **Pandas**: For data manipulation.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For building and evaluating the machine learning models.
- **NLTK**: For natural language processing and tokenization.

### Results
The model achieved strong results, especially in terms of detecting spam messages with high precision and recall. However, further improvements could be made by:

- Testing more complex models such as neural networks.
- Incorporating more advanced natural language processing techniques
