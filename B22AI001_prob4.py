import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def text_scrubber(raw_input):
    ##preprocessing
    scrubbed = str(raw_input).lower()
    scrubbed = re.sub(f"[{re.escape(string.punctuation)}]", "", scrubbed)

    scrubbed = re.sub(r'\d+', '', scrubbed)
    return scrubbed.strip()

def run_analysis_pipeline():
    source_path = 'sports_vs_politics.csv'
    
    try:
        dataset = pd.read_csv(source_path)
    except FileNotFoundError:
        print(f"Critical Error: File '{source_path}' was not found.")
        return

    # combine title and content then apply scrubbing
    dataset['processed_content'] = (dataset['title'] + " " + dataset['content']).apply(text_scrubber)
    target_labels = dataset['category']

    # 3 Feature Representations
    vectorization_methods = {
        "Bag_of_Words": CountVectorizer(stop_words='english'),
        "Bigrams": CountVectorizer(ngram_range=(2, 2), stop_words='english'),
        "TF_IDF_Weighting": TfidfVectorizer(stop_words='english')
    }

    # 3 ML Techniques
    classifier_algorithms = {
        "Multinomial_NB": MultinomialNB(),
        "Linear_SVM": SVC(kernel='linear'),
        "RF_Classifier": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    performance_log = []

    # comparison loop
    for v_name, v_tool in vectorization_methods.items():
        print(f"\n" + "#"*40)
        print(f" PROCESSING TECHNIQUE: {v_name}")
        print("#"*40)
        
        # transform text to features
        feature_matrix = v_tool.fit_transform(dataset['processed_content'])
        
        # test_size=0.20 means 20% test, 80% train.
        train_x, test_x, train_y, test_y = train_test_split(
            feature_matrix, target_labels, test_size=0.20, random_state=42, stratify=target_labels
        )

        for a_name, algorithm in classifier_algorithms.items():
            ##training and inference
            algorithm.fit(train_x, train_y)
            y_output = algorithm.predict(test_x)
            
            # confusion matrix visualization
            matrix_data = confusion_matrix(test_y, y_output)
            plt.figure(figsize=(7, 5))
            sns.heatmap(matrix_data, annot=True, fmt='g', cmap='YlGnBu', 
                        xticklabels=['Politics', 'Sport'], 
                        yticklabels=['Politics', 'Sport'])
            
            plt.title(f"Confusion Matrix: {v_name} | {a_name}")
            plt.xlabel('Predicted Category')
            plt.ylabel('True Category')
            
            img_filename = f"matrix_{v_name}_{a_name}.png"
            plt.savefig(img_filename)
            plt.close()

            current_acc = accuracy_score(test_y, y_output)
            
            print(f"\n[*] Results for {a_name}:")
            print(f"Accuracy Score: {current_acc:.4f}")
            
            print("\nDetailed Metrics:")
            print(classification_report(test_y, y_output))
            
            print("Raw Matrix:")
            print(matrix_data)
            print("-" * 25)
            
            performance_log.append({
                "Extraction_Method": v_name,
                "ML_Model": a_name,
                "Accuracy": round(current_acc, 4)
            })

    #summary
    final_report = pd.DataFrame(performance_log)
    print("\n" + "="*20 + " COMPARATIVE SUMMARY " + "="*20)
    print(final_report.sort_values(by="Accuracy", ascending=False).to_string(index=False))

if __name__ == "__main__":
    run_analysis_pipeline()
