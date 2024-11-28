# Spam_Analysis_NLP



---

# **SMS Spam Detection Analysis Using NLP**

A step-by-step detailed project explanation compiled for detecting spam messages using Natural Language Processing (NLP) techniques and machine learning models. This project involves data preprocessing, feature extraction, and building a predictive model for SMS spam classification.

---

## **Table of Contents**

1. [Project Overview](#project-overview)  
2. [Dataset Information](#dataset-information)  
3. [Stages of Analysis](#stages-of-analysis)  
4. [Technologies Used](#technologies-used)  
5. [Installation](#installation)
6. [Project Structure](#project-structure)
7. [Usage](#usage)  
8. [Results](#results)  
9. [Contributing](#contributing)  
10. [Contact](#contact)  

---

## **Project Overview**

This project applies natural language processing techniques to classify SMS messages into two categories:  
- **Spam**: Unwanted or irrelevant messages.  
- **Ham**: Legitimate messages.  

Using a dataset of 5,574 tagged SMS messages, the project demonstrates a complete workflow from data exploration to model training and evaluation.

---

## **Dataset Information**

The dataset contains SMS messages tagged as `spam` or `ham`. It is sourced from a publicly available SMS spam collection dataset.

### **Attributes**
- **Label**: The target variable indicating `spam` or `ham`.
- **Messages**: The SMS content.

---

## **Stages of Analysis**

### **1. Importing Dependencies**
Import required libraries for data handling, text preprocessing, and machine learning.

```python
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
```

---

### **2. Loading and Exploring the Dataset**
Load the dataset and inspect its structure.

```python
df = pd.read_csv('spam.csv')
df = df[['v2', 'v1']].rename(columns={'v2': 'messages', 'v1': 'label'})
df.head()
```

---

### **3. Data Cleaning and Preprocessing**
- Remove punctuation and special characters.
- Convert text to lowercase.
- Remove stop words and perform stemming.

```python
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

df['cleaned_messages'] = df['messages'].apply(preprocess_text)
```

---

### **4. Feature Extraction**
Transform text data into numerical format using **TF-IDF Vectorizer**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_messages']).toarray()
y = pd.get_dummies(df['label'], drop_first=True)  # Encode labels as binary
```

---

### **5. Model Training and Evaluation**
Train a machine learning model (e.g., Naive Bayes) on the processed data and evaluate its performance.

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

### **6. Deployment (Optional)**
Deploy the model with a Flask-based web interface to classify SMS messages in real-time.(Flask,Django,HTML,CSS,AWS)

---

## **Technologies Used**

- **Programming Language**: Python  
- **Libraries**:
  - **Data Handling**: pandas, numpy  
  - **NLP**: nltk, re  
  - **Machine Learning**: scikit-learn  
  - **Vectorization**: TfidfVectorizer  

---

## **Installation**


```bash
# Clone the repository
git clone https://github.com/vijaykumarr1452/SMS_SPAM_ANALYSIS.git

# Navigate to project directory
cd SMS_SPAM_ANALYSIS
```

---

## **Project Structure** 📂
```
SMS_SPAM_ANALYSIS/
│
├── data/
│   └── spam_dataset.csv
│
├── notebooks/
│   └── spam_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   └── model_training.py
│
├── models/
│   └── trained_models/
│
├── requirements.txt
└── README.md
```

---

## **Usage**

Run the notebook to see step-by-step analysis.

---

## **Results**

- The model achieved an **accuracy of ~97%** on the test dataset.  
- Confusion Matrix and Classification Report provide additional performance metrics.

---

## **Contributing**

1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature-name`.  
3. Commit your changes: `git commit -m 'Add new feature'`.  
4. Push to the branch: `git push origin feature-name`.  
5. Submit a pull request.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Contact**

[Linkedin](https://www.linkedin.com/in/rachuri-vijaykumar/)
[vijaykumarit45@gmail.com](mailto:vijaykumarit45@gmail.com)
[Github](https://github.com/vijaykumarr1452)
[Twitter](https://x.com/vijay_viju1)

Let me know if you'd like me to refine this further or add specific sections! 😊

---

🌟 **Star the repository if you find it helpful!** 🌟
---
