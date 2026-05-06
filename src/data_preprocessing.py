import pandas as pd

def load_train_data():
    deliveries = pd.read_csv('data/deliveries.csv')
    matches = pd.read_csv('data/matches.csv')
    return deliveries, matches


def load_test_data():
    deliveries = pd.read_csv('data/deliveries_2025.csv')
    matches = pd.read_csv('data/matches_2025.csv')
    return deliveries, matches


def preprocess_data(deliveries, matches):

    print("\nColumns in deliveries:", deliveries.columns.tolist())

    # STANDARDIZE DELIVERY DATA
    if 'match_no' in deliveries.columns:
        print("Converting 2025 dataset format → standard format")

        deliveries = deliveries.rename(columns={
            'match_no': 'match_id',
            'innings': 'inning',
            'runs_of_bat': 'batsman_runs',
            'extras': 'extra_runs'
        })

    # CREATE TOTAL RUNS
    if 'total_runs' not in deliveries.columns:
        deliveries['batsman_runs'] = pd.to_numeric(deliveries['batsman_runs'], errors='coerce').fillna(0)
        deliveries['extra_runs'] = pd.to_numeric(deliveries['extra_runs'], errors='coerce').fillna(0)
        deliveries['total_runs'] = deliveries['batsman_runs'] + deliveries['extra_runs']

    # -----------------------------
    # CREATE WICKET COLUMN
    # -----------------------------
    if 'is_wicket' not in deliveries.columns:
        if 'wicket_type' in deliveries.columns:
            deliveries['is_wicket'] = deliveries['wicket_type'].notna().astype(int)
        elif 'dismissal_kind' in deliveries.columns:
            deliveries['is_wicket'] = deliveries['dismissal_kind'].notna().astype(int)
        else:
            deliveries['is_wicket'] = 0

    deliveries['total_runs'] = deliveries['total_runs'].astype(int)
    deliveries['is_wicket'] = deliveries['is_wicket'].astype(int)

    # =============================
    # STANDARDIZE MATCHES DATA
    # =============================
    print("\nColumns in matches:", matches.columns.tolist())

    # Fix match ID column
    if 'match_no' in matches.columns:
        matches = matches.rename(columns={'match_no': 'id'})

    if 'match_id' in matches.columns and 'id' not in matches.columns:
        matches = matches.rename(columns={'match_id': 'id'})

    # Fix winner column
    if 'winner' not in matches.columns:

        print("'winner' not found → trying alternatives")

        if 'match_winner' in matches.columns:
            matches = matches.rename(columns={'match_winner': 'winner'})

        elif 'winning_team' in matches.columns:
            matches = matches.rename(columns={'winning_team': 'winner'})

        else:
            raise ValueError(f"Could not find winner column. Available: {matches.columns.tolist()}")

    # Keep only valid matches
    matches = matches[matches['winner'].notna()].copy()

    print("Deliveries shape:", deliveries.shape)
    print("Matches shape:", matches.shape)

    return deliveries, matches