import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from training_and_tunning_models.autoencoder import Autoencoder
from training_and_tunning_models.split_data import split_dataset
from training_and_tunning_models.data_preprocessing import clean_data
from training_and_tunning_models.parametertuning_table import generate_ac_parameter_table

def ac_parameter_tuning():
    # Split dataset into train, validation, and test sets
    train_df, val_df, test_df, spam_emails_df = split_dataset('datasets/SpamAssasin.csv')
    # train_df, val_df, test_df, spam_emails_df = split_dataset('datasets/CEAS_08.csv')

    # Clean the data
    train, train_labels = clean_data(train_df)
    test, test_labels = clean_data(test_df)

    # Split train data into train and validation sets
    x_non_spam_train_data, x_non_spam_val_data = train_test_split(train, test_size=0.25, random_state=42)

    # Convert the data to TF-IDF features
    vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    x_non_spam_train_data = vectorizer.fit_transform(x_non_spam_train_data).toarray()
    x_non_spam_test_data = vectorizer.transform(x_non_spam_val_data).toarray()

    # Convert the spam emails to TF-IDF features for evaluation
    train = vectorizer.transform(train).toarray()
    spam_emails_df = vectorizer.transform(spam_emails_df).toarray()

    # Convert validation test and test data to TF-IDF features
    test_data = vectorizer.transform(test).toarray()

    # Convert to PyTorch tensors
    x_non_spam_train_tensor = torch.tensor(x_non_spam_train_data, dtype=torch.float32)
    x_non_spam_val_tensor = torch.tensor(x_non_spam_test_data, dtype=torch.float32)
    x_only_non_spam_data_tensor = torch.tensor(train, dtype=torch.float32)
    x_only_spam_data_tensor = torch.tensor(spam_emails_df, dtype=torch.float32)
    x_test_data_tensor = torch.tensor(test_data, dtype=torch.float32)

    # Define hyperparameter grid
    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "encoding_dim": [8, 16, 32, 64],
        "num_epochs": [20, 30, 35, 40],
        "optimizer": ["adam", "sgd", "rmsprop"]
    }

    # Initialize variables to track the best configuration
    best_params = None
    best_threshold = None
    best_f1 = 0
    best_val_loss = float('inf')
    all_results = []

    # Calculate total combinations
    total_combinations = len(param_grid["learning_rate"]) * len(param_grid["encoding_dim"]) * len(param_grid["num_epochs"] * len(param_grid["optimizer"]))
    current_combination = 0

    # Perform grid search
    for lr in param_grid["learning_rate"]:
        for encoding_dim in param_grid["encoding_dim"]:
            for num_epochs in param_grid["num_epochs"]:
                for opt_name in param_grid["optimizer"]:
                    current_combination += 1
                    remaining_combinations = total_combinations - current_combination

                    print(f"Testing Parameters: learning_rate={lr}, encoding_dim={encoding_dim}, num_epochs={num_epochs}, optimizer={opt_name}")
                    print(f"Remaining Combinations: {remaining_combinations} \n")

                    input_dim = x_non_spam_train_tensor.shape[1]
                    model = Autoencoder(input_dim, encoding_dim)
                    criterion = nn.MSELoss()

                    # Optimizer switch
                    if opt_name == "adam":
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                    elif opt_name == "sgd":
                        optimizer = optim.SGD(model.parameters(), lr=lr)
                    elif opt_name == "rmsprop":
                        optimizer = optim.RMSprop(model.parameters(), lr=lr)
                    else:
                        raise ValueError(f"Unknown optimizer: {opt_name}")
                    # Train the model
                    model.train()
                    for epoch in range(num_epochs):
                        optimizer.zero_grad()
                        output = model(x_non_spam_train_tensor)
                        loss = criterion(output, x_non_spam_train_tensor)
                        loss.backward()
                        optimizer.step()

                    # Evaluate on validation set
                    model.eval()
                    with torch.no_grad():
                        val_reconstructions = model(x_non_spam_val_tensor)
                        val_errors = torch.mean((val_reconstructions - x_non_spam_val_tensor) ** 2, dim=1).numpy()
                        val_loss = np.mean(val_errors)
                        print(f"Validation Loss: {np.mean(val_errors):.4f}")


                    if val_loss <= 0.0100:
                        print(f"Skipping parameters due to too-low validation loss: {val_loss:.4f}")
                        continue

                    # Save results for this parameter set
                    all_results.append({
                        "learning_rate": lr,
                        "encoding_dim": encoding_dim,
                        "num_epochs": num_epochs,
                        "optimizer": opt_name,
                        "val_loss": np.mean(val_errors)
                    })
                    # Update global best if this is the best so far
                    val_loss = np.mean(val_errors)
                    if best_params is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {
                            "learning_rate": lr,
                            "encoding_dim": encoding_dim,
                            "num_epochs": num_epochs,
                            "optimizer": opt_name,
                            "val_loss": val_loss
                        }

    # Print the best parameters and validation loss
    print(f"Best Parameters: {best_params}")
    # print(f"Best Validation Loss: {best_val_loss:.4f}")

    # === Train and save the best model ===
    # Re-initialize the best model with best parameters
    input_dim = x_non_spam_train_tensor.shape[1]
    best_model = Autoencoder(input_dim, best_params["encoding_dim"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_params["learning_rate"])

    # Train the best model on the full training set
    best_model.train()
    for epoch in range(best_params["num_epochs"]):
        optimizer.zero_grad()
        output = best_model(x_non_spam_train_tensor)
        loss = criterion(output, x_non_spam_train_tensor)
        loss.backward()
        optimizer.step()

    # Save the trained best model
    torch.save(best_model.state_dict(), "models/autoencoder_best_model.pt")
    print("Best autoencoder model saved to autoencoder_best_model.pt")

    # Testing the best model on the test set
    best_model.eval()
    with torch.no_grad():
        reconstructions = best_model(x_test_data_tensor)
        errors = torch.mean((reconstructions - x_test_data_tensor) ** 2, dim=1).numpy()

    thresholds = np.linspace(errors.min(), errors.max(), 100)
    best_f1 = 0
    best_threshold = None
    best_precision = 0
    best_recall = 0

    # Try multiple thresholds
    for t in thresholds:
        preds = (errors > t).astype(int)
        f1 = f1_score(test_labels, preds)
        precision = precision_score(test_labels, preds)
        recall = recall_score(test_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall

    print(f"Best Parameters: {best_params}, Best F1: {best_f1}")
    print(f"Best Validation Loss: {best_val_loss:.4f}, reconstruction error: {errors}")

    with open("models/autoencoder_param_tuning_results.pkl", "wb") as f:
        pickle.dump({
            "all_results": all_results,
            "best_params": best_params,
            "best_f1": best_f1
        }, f)

    print("All parameter tuning results saved to autoencoder_param_tuning_results.pkl")
    generate_ac_parameter_table()

