import matplotlib.pyplot as plt
import pandas as pd


# HISTORICAL (2008–2024)

def plot_powerplay_vs_win(df):
    plt.figure()
    plt.scatter(df['team2_pp_runs'], df['team2_win'])
    plt.xlabel("Team2 Powerplay Runs")
    plt.ylabel("Win (1) / Loss (0)")
    plt.title("Powerplay Runs vs Win Outcome (2008–2024)")
    plt.savefig('outputs/historical_pp_vs_win.png')
    plt.close()


def plot_powerplay_distribution(df):
    plt.figure()
    plt.hist(df['team2_pp_runs'], bins=20)
    plt.title("Powerplay Runs Distribution (2008–2024)")
    plt.xlabel("Runs")
    plt.ylabel("Frequency")
    plt.savefig('outputs/historical_pp_distribution.png')
    plt.close()


def plot_wickets_vs_win(df):
    plt.figure()
    plt.scatter(df['team2_pp_wickets'], df['team2_win'])
    plt.xlabel("Wickets Lost in Powerplay")
    plt.ylabel("Win (1/0)")
    plt.title("Wickets vs Win Outcome")
    plt.savefig('outputs/historical_wickets_vs_win.png')
    plt.close()


def plot_run_distribution(df):
    plt.figure()
    plt.hist(df['target_runs'], bins=20)
    plt.title("Target Score Distribution (2008–2024)")
    plt.xlabel("Runs")
    plt.ylabel("Frequency")
    plt.savefig('outputs/historical_target_distribution.png')
    plt.close()


# 2025 PREDICTION ANALYSIS

def plot_prediction_vs_actual(results):
    correct = (results['winner'] == results['predicted_winner']).sum()
    incorrect = len(results) - correct

    plt.figure()
    plt.bar(['Correct', 'Incorrect'], [correct, incorrect])
    plt.title("Prediction Accuracy (2025)")
    plt.savefig('outputs/2025_accuracy_bar.png')
    plt.close()


def plot_probability_distribution(results):
    plt.figure()
    plt.hist(results['win_probability_team2'], bins=20)
    plt.title("Win Probability Distribution (2025)")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.savefig('outputs/2025_probability_distribution.png')
    plt.close()


def plot_probability_vs_result(results):
    plt.figure()
    plt.scatter(results['win_probability_team2'], results['team2_win'])
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Result")
    plt.title("Probability vs Actual Outcome")
    plt.savefig('outputs/2025_probability_vs_actual.png')
    plt.close()


def plot_powerplay_vs_probability(results):
    plt.figure()
    plt.scatter(results['team2_pp_runs'], results['win_probability_team2'])
    plt.xlabel("Powerplay Runs")
    plt.ylabel("Win Probability")
    plt.title("Powerplay vs Win Probability (2025)")
    plt.savefig('outputs/2025_pp_vs_probability.png')
    plt.close()