Project Overview
----------------
This project implements a Natural Language Processing (NLP) pipeline designed to classify text documents into two categories: Sport and Politics. The goal is to evaluate how different feature extraction techniques interact with multiple machine learning algorithms to achieve optimal classification accuracy.

Dataset
-------
The project uses the BBC News Archive dataset obtained from Kaggle.

Source: https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive

Modifications:
- Filtered to keep only the Sport and Politics categories.
- Final dataset size: 928 documents.
- Title and body text were merged into one combined text field.
- An 80/20 stratified train-test split was used to preserve class balance.

Feature Extraction Methods
--------------------------
1. Bag of Words (BoW)
   - Represents text using raw word frequency counts.

2. n-grams (Bigrams)
   - Captures two-word sequences to retain local context patterns.

3. TF-IDF
   - Weights words based on importance relative to the entire corpus,
     reducing the influence of common but uninformative terms.

Machine Learning Models
-----------------------
1. Multinomial Naive Bayes
   - A probabilistic classifier effective for sparse, high-dimensional text.

2. Linear Support Vector Machine (SVM)
   - Finds an optimal separating hyperplane through margin maximization.

3. Random Forest
   - An ensemble of decision trees that aggregates predictions.

Results
-------
Accuracy scores for all nine feature-model combinations:

Feature Extraction        Classifier Model      Accuracy
--------------------------------------------------------
Bag of Words              Naive Bayes           1.0000
n-grams (Bigrams)         Naive Bayes           1.0000
TF-IDF                    Naive Bayes           1.0000
TF-IDF                    SVM (Linear)          1.0000
Bag of Words              SVM (Linear)          0.9946
Bag of Words              Random Forest         0.9892
TF-IDF                    Random Forest         0.9839
n-grams (Bigrams)         SVM (Linear)          0.9247
n-grams (Bigrams)         Random Forest         0.9247

Key Findings
------------
- Naive Bayes achieved perfect accuracy across all feature types,
  indicating that the vocabularies of the two categories are highly distinguishable.

- Bigrams reduced performance for SVM and Random Forest because of sparsity:
  many word-pairs appeared too infrequently to form reliable patterns.

- TF-IDF effectively filtered out high-frequency noise, enabling the Linear SVM
  to also achieve 100% accuracy.

