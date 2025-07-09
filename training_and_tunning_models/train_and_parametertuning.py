# Importing the necessary modules
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from training_and_tunning_models.data_preprocessing import clean_data
from training_and_tunning_models.parametertuning_table import generate_models_parameter_table
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier


# Define models and their hyperparameter grids 💫
models_and_parameters = {
    "Naive Bayes": (MultinomialNB(), {
        'alpha': [0.01, 0.1, 0.5, 1.0, 10]
    }),
    "Random Forest": (RandomForestClassifier(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2']
    }),
    "Gradient Boost": (GradientBoostingClassifier(), {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10]
    }),
    "AdaBoost": (AdaBoostClassifier(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }),
    "Logistic Regression": (LogisticRegression(), {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100, 200, 500]
    }),
    "KNN": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    })
}


def train_model():
    import pickle
    
    # Load and clean the dataset 
    x, y = clean_data('datasets/SpamAssasin.csv')

    # vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    x_vectorized = vectorizer.fit_transform(x)

    # Save the vectorizer for later use
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Split your dataset using the vectorized features
    x_train_vec, x_test_vec, y_train, y_test = train_test_split(
        x_vectorized, y, test_size=0.1, random_state=225, stratify=y
    ) # splited the dataset into 90% training and 10% testing

    # Store the best models after tuning ✨
    best_models = {}
    all_results = []

    # Calculate total rounds
    total_rounds = len(models_and_parameters)
    current_round = 0

    for name, (model, params) in models_and_parameters.items():
        current_round += 1
        rounds_left = total_rounds - current_round
        print(f"\n💫 Tuning and training: {name}")
        print(f"Rounds left: {rounds_left}")
        grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
        grid.fit(x_train_vec, y_train)
        
        best_models[name] = grid.best_estimator_
        
        # print(f"✅ Best Params for {name}: {grid.best_params_}")
        y_pred = grid.predict(x_test_vec)
        report = classification_report(y_test, y_pred, output_dict=True)
        # print(f"📊 Performance for {name}:\n{classification_report(y_test, y_pred)}")
        
        # Extract mean validation accuracy for each parameter combination
        mean_val_scores = grid.cv_results_['mean_test_score']
        params_list = grid.cv_results_['params']
        param_scores = [
            {"params": params, "mean_val_accuracy": score}
            for params, score in zip(params_list, mean_val_scores)
        ]

        all_results.append({
            "model": name,
            "best_params": grid.best_params_,
            "cv_results": grid.cv_results_,
            "param_scores": param_scores,
            "classification_report": report
        })

    # Save best models to a file
    with open("models/best_models.pkl", "wb") as f:
        pickle.dump(best_models, f)
    print("Best trained models saved to best_models.pkl")

    # Save results to a file
    import pickle
    with open("models/all_model_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("All results and parameters saved to all_model_results.pkl")

    generate_models_parameter_table()