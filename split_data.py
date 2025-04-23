import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(dataset_path):
    # Importing the dataset
    df = pd.read_csv(dataset_path)

    # Splitting the dataset into spam and non-spam emails
    non_spam_emails_df  = df[df['label']== 0]
    spam_emails_df = df[df['label'] == 1]

    # print amount of spam and non-spam emails
    print(f"Spam emails: {len(spam_emails_df)}")
    print(f"Non-spam emails: {len(non_spam_emails_df)}")

    # Creating training set of only the non-spam emails
    train_df = non_spam_emails_df.sample(n=int(len(non_spam_emails_df) * 0.9), random_state=42)

    # Adding remaining non-spam emails to remaining_non_spam_emails
    remaining_non_spam_emails = non_spam_emails_df.drop(train_df.index)

    # Sample 212 spam emails and 212 non spam emails for validation
    val_spam_emails = spam_emails_df.sample(n=int(len(spam_emails_df) / 2), random_state=42)
    spam_emails_df_remaining = spam_emails_df.drop(val_spam_emails.index)

    val_non_spam_emails = remaining_non_spam_emails.sample(n=int(len(remaining_non_spam_emails) / 2), random_state=1)
    remaining_non_spam_emails = remaining_non_spam_emails.drop(val_non_spam_emails.index)

    val_df = pd.concat([val_spam_emails, val_non_spam_emails])
    val_df = val_df.sample(frac=1, random_state=1).reset_index(drop=True)

    # Sample 212 nonspam emails for testing
    test_spam_emails = spam_emails_df_remaining.sample(n=int(len(spam_emails_df_remaining)), random_state=2)
    test_non_spam_emails = remaining_non_spam_emails.sample(n=int(len(remaining_non_spam_emails)), random_state=2)

    test_df = pd.concat([test_spam_emails, test_non_spam_emails])
    test_df = test_df.sample(frac=1, random_state=1).reset_index(drop=True)

    # Checking the data
    print(f"Train set: {len(train_df)} (Non-Spam only)")
    print(f"Validation set: {len(val_df)} ({val_df['label'].value_counts().to_dict()})")
    print(f"Test set: {len(test_df)} ({test_df['label'].value_counts().to_dict()})")

    return train_df, val_df, test_df, spam_emails_df