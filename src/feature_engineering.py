import pandas as pd

def create_features(deliveries, matches):

    # -----------------------------
    # FIRST INNINGS POWERPLAY
    # -----------------------------
    pp1 = deliveries[(deliveries['inning'] == 1) & (deliveries['over'] <= 6)]

    team1_pp = pp1.groupby('match_id').agg({
        'total_runs': 'sum',
        'is_wicket': 'sum'
    }).reset_index()

    team1_pp.rename(columns={
        'total_runs': 'team1_pp_runs',
        'is_wicket': 'team1_pp_wickets'
    }, inplace=True)

    # -----------------------------
    # SECOND INNINGS POWERPLAY
    # -----------------------------
    pp2 = deliveries[(deliveries['inning'] == 2) & (deliveries['over'] <= 6)]

    team2_pp = pp2.groupby('match_id').agg({
        'total_runs': 'sum',
        'is_wicket': 'sum'
    }).reset_index()

    team2_pp.rename(columns={
        'total_runs': 'team2_pp_runs',
        'is_wicket': 'team2_pp_wickets'
    }, inplace=True)

    # -----------------------------
    # FIRST INNINGS TOTAL (TARGET)
    # -----------------------------
    target = deliveries[deliveries['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    target.rename(columns={'total_runs': 'target_runs'}, inplace=True)

    # -----------------------------
    # MERGE BASE DATA
    # -----------------------------
    df = team1_pp.merge(team2_pp, on='match_id')
    df = df.merge(target, on='match_id')

    df = df.merge(
        matches[['id', 'winner', 'team1', 'team2', 'venue', 'toss_decision']],
        left_on='match_id',
        right_on='id'
    )

    # -----------------------------
    # TARGET VARIABLE
    # -----------------------------
    df['team2_win'] = (df['winner'] == df['team2']).astype(int)

    # =============================
    # 🔥 NEW FEATURES START HERE
    # =============================

    # 1. Momentum
    df['pp_run_diff'] = df['team2_pp_runs'] - df['team1_pp_runs']

    # 2. Pressure
    df['runs_remaining'] = df['target_runs'] - df['team2_pp_runs']
    df['required_rr'] = df['target_runs'] / 20

    # 3. Wicket pressure
    df['wicket_diff'] = df['team2_pp_wickets'] - df['team1_pp_wickets']

    # -----------------------------
    # VENUE FEATURES
    # -----------------------------

    # Fix for datasets without target_runs
    if 'target_runs' not in matches.columns:
        print("⚠️ target_runs not found → using first_ings_score")

        if 'first_ings_score' in matches.columns:
            matches['target_runs'] = matches['first_ings_score']
        else:
            raise ValueError("❌ No column available for target runs")

    # Now safe to compute venue averages
    venue_avg = matches.groupby('venue')['target_runs'].mean()

    df['venue_avg_score'] = df['venue'].map(venue_avg)


    df['high_scoring_ground'] = (df['venue_avg_score'] > 170).astype(int)

    # -----------------------------
    # TOSS / DEW PROXY
    # -----------------------------
    df['toss_chose_bowl'] = (df['toss_decision'] == 'field').astype(int)

    # -----------------------------
    # TEAM STRENGTH
    # -----------------------------
    team_wins = matches['winner'].value_counts()
    team_matches = matches['team1'].value_counts() + matches['team2'].value_counts()

    team_strength = (team_wins / team_matches).fillna(0)

    df['team1_strength'] = df['team1'].map(team_strength).fillna(0)
    df['team2_strength'] = df['team2'].map(team_strength).fillna(0)

    # -----------------------------
    # CLEANUP
    # -----------------------------
    df = df.drop(columns=['id'])

    return df