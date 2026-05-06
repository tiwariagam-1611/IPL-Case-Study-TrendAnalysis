from src.data_preprocessing import load_train_data, load_test_data, preprocess_data
from src.feature_engineering import create_features
from src.train_model import train_model
from src.predict_2025 import predict_2025
from src.visualization import *

def main():

    # ---------------- TRAIN ----------------
    print("Loading TRAIN data...")
    d_train, m_train = load_train_data()
    d_train, m_train = preprocess_data(d_train, m_train)

    print("Creating TRAIN features...")
    train_df = create_features(d_train, m_train)

    print("Training model...")
    model = train_model(train_df)

    print("Generating historical visualizations...")

    plot_powerplay_vs_win(train_df)
    plot_powerplay_distribution(train_df)
    plot_wickets_vs_win(train_df)
    plot_run_distribution(train_df)

    # ---------------- TEST (2025) ----------------
    print("Loading TEST data (2025)...")
    d_test, m_test = load_test_data()
    d_test, m_test = preprocess_data(d_test, m_test)

    print("Creating TEST features...")
    test_df = create_features(d_test, m_test)

    print("Predicting 2025 matches...")
    results = predict_2025(model, test_df)

    print(results[['team1', 'team2', 'winner', 'predicted_winner', 'win_probability_team2']].head())

    print("\nPredictions saved in outputs/2025_predictions.csv")

    print("Generating 2025 visualizations...")

    plot_prediction_vs_actual(results)
    plot_probability_distribution(results)
    plot_probability_vs_result(results)
    plot_powerplay_vs_probability(results)


if __name__ == "__main__":
    main()