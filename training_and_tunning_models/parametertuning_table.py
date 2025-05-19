import pickle
import pandas as pd


def generate_ac_parameter_table():
    with open("models/autoencoder_param_tuning_results.pkl", "rb") as f:
        ac_results = pickle.load(f)

    # Show all parameter combinations and their validation loss
    df = pd.DataFrame(ac_results["all_results"])
    print("All Autoencoder Parameter Tuning Results:")
    print(df)

    # Save all autoencoder results as CSV and Excel
    df.to_csv("tables/autoencoder_param_tuning_results_table.csv", index=False)
    df.to_excel("tables/autoencoder_param_tuning_results_table.xlsx", index=False)

    print("All autoencoder results saved to tables/autoencoder_param_tuning_results_table.csv and tables/autoencoder_param_tuning_results_table.xlsx")

def generate_models_parameter_table():
    # Show the best model
    with open("models/all_model_results.pkl", "rb") as f:
        all_results = pickle.load(f)

    # Show best parameters for each model
    rows = []
    for result in all_results:
        row = {"model": result["model"]}
        row.update(result["best_params"])
        rows.append(row)
    df2 = pd.DataFrame(rows)
    print("\nBest Parameters for Each Model:")
    print(df2)

    # Save all model results as CSV and Excel
    df2.to_csv("tables/all_model_results_table.csv", index=False)
    df2.to_excel("tables/all_model_results_table.xlsx", index=False)

    print("All model results saved to tables/all_model_results_table.csv and tables/all_model_results_table.xlsx")