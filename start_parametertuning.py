from training_and_tunning_models.train_and_parametertuning import train_model
from training_and_tunning_models.ac_parameter_tuning import ac_parameter_tuning


def main():
    # Train and save the best model
    train_model()

    # Perform parameter tuning for the autoencoder
    ac_parameter_tuning()

    print("Parameter tuning completed.")

if __name__ == "__main__":
    main()

