# 📩 SMS Spam Detection using Machine Learning

This project was developed as part of my internship at **CodSoft**, where I worked as a **Machine Learning Intern**. The goal was to build a machine learning model capable of detecting and classifying SMS messages as **spam** or **ham** (not spam) using natural language processing (NLP) techniques.

---

## 🧠 Problem Statement

SMS spam is a growing problem that affects millions of users worldwide. The objective of this project is to create an intelligent classifier that can filter out spam messages, thereby improving user experience and security.

---

## 📊 Dataset

- **Source**: [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Total Messages**: ~5,500
- **Classes**:
  - `ham`: Legitimate messages
  - `spam`: Unwanted messages

---

## 🔧 Project Workflow

### 1. **Data Preprocessing**
- Lowercased the text
- Removed punctuation and stopwords
- Tokenization and stemming

### 2. **Feature Extraction**
- Used **TF-IDF Vectorizer** to convert text into numerical features

### 3. **Model Building**
- Trained several models including:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Logistic Regression

### 4. **Evaluation Metrics**
- **Accuracy**
- **Precision & Recall**
- **F1-Score**
- **Confusion Matrix**

---

## 🧪 Results

| Model              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Naive Bayes        | 98.6%    | 96%       | 95%    | 95.5%    |
| SVM                | 97.9%    | 94%       | 93%    | 93.5%    |

✅ **Naive Bayes** performed best due to its strength in handling text classification problems.

---

## 📽️ Demo Video

Watch the short demonstration here:  
👉 *[Attach your compressed `spam_compressed.mp4` video here]*

---

## 🧰 Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Matplotlib & Seaborn
- Jupyter Notebook

---

## 📌 Key Learnings

- Preprocessing text data is crucial for accurate classification
- Naive Bayes is highly effective for NLP tasks like spam detection
- Evaluation metrics must go beyond accuracy in imbalanced datasets

---

## 🔗 Connect With Me

> 💼 This project is a part of my internship at **CodSoft** under the role of **Machine Learning Intern**.  
> I'm always open to feedback and collaboration opportunities!

#SMSDetection #SpamClassifier #MachineLearning #CodSoftInternship #NLP #Python #MLProject #TextClassification #LinkedInProjects
