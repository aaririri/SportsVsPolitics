import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def clean_text(text):
    #basic preprocessing 
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = text.strip()
    return text

def main():
    # 1. Load the CSV we generated in the previous step
    input_file = 'sports_vs_politics.csv'
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run your sep.py script first!")
        return

    # 2. Preprocess
    # Assuming the columns are 'category' and 'title' or 'content'
    # Let's combine title and content for better feature depth
    if 'title' in df.columns and 'content' in df.columns:
        df['combined_text'] = df['title'] + " " + df['content']
    else:
        # Fallback if your CSV has different names
        df['combined_text'] = df.iloc[:, 1] 

    df['clean_text'] = df['combined_text'].apply(clean_text)

    # 3. Feature Representation: TF-IDF with Unigrams and Bigrams
    # ngram_range=(1, 2) helps capture phrases like "prime minister" or "home run"
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['category']

    # 4. Split data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 5. Define the 3 required ML Techniques
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine": SVC(kernel='linear', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # 6. Train and Compare
    print(f"Total Samples: {len(df)} | Features: {X.shape[1]}")
    print("-" * 40)
    
    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        results[name] = acc
        
        print(f"Model: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, predictions))
        print("-" * 40)

    # Summary table for your report
    print("\nFINAL COMPARISON TABLE")
    print(f"{'Model':<25} | {'Accuracy':<10}")
    for name, acc in results.items():
        print(f"{name:<25} | {acc:.4f}")

if __name__ == "__main__":
    main()