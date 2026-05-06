from sklearn.ensemble import GradientBoostingClassifier

def train_model(df):

    X = df[
        [
            # base
            'team1_pp_runs',
            'team1_pp_wickets',
            'team2_pp_runs',
            'team2_pp_wickets',
            'target_runs',

            # new
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

    y = df['team2_win']

    model = GradientBoostingClassifier()
    model.fit(X, y)

    return model