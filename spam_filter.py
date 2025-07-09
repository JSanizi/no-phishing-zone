import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from application.ac_filter import run_autoencoder
from application.sending_mail import sending_emails
from application.connect_to_email import get_unread_emails
from sklearn.metrics import classification_report, confusion_matrix
from training_and_tunning_models.data_preprocessing import clean_data

# Load vectorizer and models at the top
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
with open("models/best_models.pkl", "rb") as f:
    best_models = pickle.load(f)
 

def classify_with_all_models(email_texts):
    # email_texts should be a list of raw emails
    for model_name in best_models.keys():
        pred_file = f"email_labels/{model_name}_predicted_sent_emails.csv"
        # If the file exists, delete it so we start fresh
        if os.path.exists(pred_file):
            os.remove(pred_file)

    for idx, email_text in enumerate(email_texts):
        # print(f"\n💌 Classifying email #{idx+1}")

        # Clean and vectorize
        cleaned = clean_data(email_text)
        vec = vectorizer.transform(cleaned)

        # Go through each model and show the prediction
        for model_name, model in best_models.items():
            try:
                pred = model.predict(vec)[0]
            except Exception as e:
                print(f"\n👑 Using {model_name} model:")
                print(f"🚨 Prediction failed: {e}")
                continue
            try:
                class_idx = list(model.classes_).index(pred)
                prob = model.predict_proba(vec)[0][class_idx]
            except Exception:
                prob = "N/A"

            # Append the prediction to the file (create new if not exists)
            pred_file = f"email_labels/{model_name}_predicted_sent_emails.csv"
            write_header = not os.path.exists(pred_file)
            with open(pred_file, "a") as f:
                if write_header:
                    f.write("idx,prediction\n")
                if pred == 1:
                    f.write(f"{idx},Spam\n")
                else:
                    f.write(f"{idx},Non-Spam\n")

            label = "⚠️ SPAM ⚠️" if pred == 1 else "💌 NOT SPAM 💌"
            # print(f"\n👑 Using {model_name} model:")
            # print(f"👉 Prediction: {label}")
            # print(f"🎯 Confidence: {prob if prob != 'N/A' else 'Not Available'}")
    
    # Function to run the autoencoder model
    run_autoencoder(email_texts)

    # Function to print confusion matrix and classification report for all models
    printing_confusion_matrix()
    plt.show()



def printing_confusion_matrix():
    # Load true labels
    true_labels = []
    with open('email_labels/true_sent_emails.csv', 'r') as f:
        next(f)  # Skip header
        for line in f:
            true_labels.append(line.strip().split(',')[1])  # Assuming the second column is the label

    # For each model, load predicted labels and print confusion matrix
    for model_name in best_models.keys():
        predicted_labels = []
        with open(f"email_labels/{model_name}_predicted_sent_emails.csv", 'r') as f:
            next(f)  # Skip header
            for line in f:
                predicted_labels.append(line.strip().split(',')[1])  # Assuming the second column is the label

        # print(f"\n📊 Confusion Matrix for {model_name}:")
        # print(confusion_matrix(true_labels, predicted_labels))
        # print(f"\n📈 Classification Report for {model_name}:")
        # print(classification_report(true_labels, predicted_labels))

        # Save the confusion matrix as an image
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
        plt.title(f'Confusion Matrix for {model_name} with {len(true_labels)} emails\n')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if os.path.exists(f'graphs/confusion/{len(true_labels)}_emails/{model_name}_confusion_matrix.png'):
            os.remove(f'graphs/confusion/{len(true_labels)}_emails/{model_name}_confusion_matrix.png')
        plt.savefig(f'graphs/confusion/{len(true_labels)}_emails/{model_name}_confusion_matrix.png')
        plt.close()
        print(f"📸 Confusion matrix saved as {model_name}_confusion_matrix.png")
    
        # Save the classification report as a text file
        if os.path.exists(f'graphs/classification_reports/{len(true_labels)}_emails/{model_name}_classification_report.txt'):
            os.remove(f'graphs/classification_reports/{len(true_labels)}_emails/{model_name}_classification_report.txt')

        report = classification_report(true_labels, predicted_labels, target_names=['Non-Spam', 'Spam'], zero_division=0)
        with open(f'graphs/classification_reports/{len(true_labels)}_emails/{model_name}_classification_report.txt', 'w') as f:
            f.write(f'Amount of emails: {len(true_labels)}\n')
            f.write(report)
        print(f"📄 Classification report saved as {model_name}_classification_report.txt")



def main():
    # Starting with sending emails
    sending_emails()

    # Fetch and classify all emails
    raw_emails = get_unread_emails()

    # If there are unread emails, classify them
    if raw_emails:
        classify_with_all_models(raw_emails)
        print("📬 Emails processed successfully.")
    else:
        print("📭 No email to process.")

if __name__ == "__main__":
    main()