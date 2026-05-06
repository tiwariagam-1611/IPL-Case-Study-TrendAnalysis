import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    brier_score_loss
)


def predict_2025(model, test_df):

    X_test = test_df[
    [
        'team1_pp_runs',
        'team1_pp_wickets',
        'team2_pp_runs',
        'team2_pp_wickets',
        'target_runs',

        'pp_run_diff',
        'runs_remaining',
        'required_rr',
        'wicket_diff',

        'venue_avg_score',
        'high_scoring_ground',
        'toss_chose_bowl',

        'team1_strength',
        'team2_strength'
    ]
    ]

    y_true = test_df['team2_win']

    # Predictions
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Add results
    results = test_df.copy()
    results['predicted_team2_win'] = preds
    results['win_probability_team2'] = probs
    results['team1_win_probability'] = 1 - probs
    results['predicted_winner'] = results.apply(
        lambda row: row['team2'] if row['predicted_team2_win'] == 1 else row['team1'],
        axis=1
    )

    # METRICS

    accuracy = accuracy_score(y_true, preds)
    conf_matrix = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds)

    logloss = log_loss(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)

    # PRINT RESULTS

    print("\n================ 2025 MODEL PERFORMANCE ================\n")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(report)

    print("=======================================================\n")

    # Save results
    results.to_csv('outputs/2025_predictions.csv', index=False)

    return results