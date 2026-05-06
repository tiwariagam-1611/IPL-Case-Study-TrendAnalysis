from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]  # probability of team2 winning

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print("\nModel Performance")
    print("Accuracy:", acc)
    print(report)

    # Create result dataframe
    results_df = X_test.copy()
    results_df['actual'] = y_test.values
    results_df['predicted'] = preds
    results_df['win_probability_team2'] = probs

    # Save results
    results_df.to_csv('outputs/predictions.csv', index=False)

    with open('outputs/results.txt', 'w') as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)

    return acc, results_df