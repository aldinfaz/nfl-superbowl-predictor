import pandas as pd
import numpy as np
import matplotlib as plt

file_path = '/Users/aldinfazlic/Desktop/projects/nfl_v2/nfl_team_stats_2002-2023.csv'
df = pd.read_csv(file_path)

def pos_to_seconds(pos):
    minutes, seconds = pos.split(':')
    total = (int(minutes) * 60 ) + int(seconds)

    return total

df['possession_away_seconds'] = df['possession_away'].apply(pos_to_seconds)
df['possession_home_seconds'] = df['possession_home'].apply(pos_to_seconds)

df['home_win'] = (df['score_home'] > df['score_away']).astype(int)
df['home_loss'] = (df['score_home'] < df['score_away']).astype(int)

# home team stats (ex. ravens stats when home)
df_home = df.groupby(['season', 'home']).agg(
    total_pts=('score_home', 'sum'),
    total_pts_against=('score_away', 'sum'),
    total_ints_off=('interceptions_home', 'sum'),
    total_ints_def=('interceptions_away', 'sum'),
    total_third_downs=('third_down_att_home', 'sum'),
    total_completed_third_downs=('third_down_comp_home', 'sum'),
    wins=('home_win', 'sum'),
    losses=('home_loss', 'sum')
).reset_index()

# away team stats (ex. ravens stats when away)
df_away = df.groupby(['season', 'away']).agg(
    total_pts=('score_away', 'sum'),
    total_pts_against=('score_home', 'sum'),
    total_ints_off=('interceptions_away', 'sum'),
    total_ints_def=('interceptions_home', 'sum'),
    total_third_downs=('third_down_att_away', 'sum'),
    total_completed_third_downs=('third_down_comp_away', 'sum'),
    wins=('home_loss', 'sum'),  # A home loss is an away win
    losses=('home_win', 'sum')  # A home win is an away loss
).reset_index()

# Rename 'away' to 'home' to match columns
df_away = df_away.rename(columns={'away': 'home'})

# Combine home and away stats by summing them
dftotal = pd.merge(df_home, df_away, on=['season', 'home'], how='outer', suffixes=('_home', '_away'))

# calculate total stats
dftotal['total_pts'] = dftotal['total_pts_home'].fillna(0) + dftotal['total_pts_away'].fillna(0)
dftotal['total_pts_against'] = dftotal['total_pts_against_home'].fillna(0) + dftotal['total_pts_against_away'].fillna(0)
dftotal['total_ints_off'] = dftotal['total_ints_off_home'].fillna(0) + dftotal['total_ints_off_away'].fillna(0)
dftotal['total_ints_def'] = dftotal['total_ints_def_home'].fillna(0) + dftotal['total_ints_def_away'].fillna(0)
dftotal['total_third_downs'] = dftotal['total_third_downs_home'].fillna(0) + dftotal['total_third_downs_away'].fillna(0)
dftotal['total_completed_third_downs'] = dftotal['total_completed_third_downs_home'].fillna(0) + dftotal['total_completed_third_downs_away'].fillna(0)
dftotal['wins'] = dftotal['wins_home'].fillna(0) + dftotal['wins_away'].fillna(0)
dftotal['losses'] = dftotal['losses_home'].fillna(0) + dftotal['losses_away'].fillna(0)
dftotal['games_played'] = dftotal['wins'] + dftotal['losses']

# calculate average stats
dftotal['avg_pts'] = dftotal['total_pts'] / dftotal['games_played']
dftotal['third_down_conversion'] = dftotal['total_completed_third_downs'] / dftotal['total_third_downs']

#if team won superbowl that year
superbowl_winners = [{'season': 2002, 'winning_team': 'Buccaneers'},
    {'season': 2003, 'winning_team': ' Patriots'},
    {'season': 2004, 'winning_team': 'Patriots'},
    {'season': 2005, 'winning_team': 'Steelers'},
    {'season': 2006, 'winning_team': 'Steelers'},
    {'season': 2007, 'winning_team': 'Colts'},
    {'season': 2008, 'winning_team': 'New York Giants'},
    {'season': 2009, 'winning_team': 'Steelers'},
    {'season': 2010, 'winning_team': 'Saints'},
    {'season': 2011, 'winning_team': 'Packers'},
    {'season': 2012, 'winning_team': 'New York Giants'},
    {'season': 2013, 'winning_team': 'Ravens'},
    {'season': 2014, 'winning_team': 'Patriots'},
    {'season': 2015, 'winning_team': 'Broncos'},
    {'season': 2016, 'winning_team': 'Patriots'},
    {'season': 2017, 'winning_team': 'Eagles'},
    {'season': 2018, 'winning_team': 'Patriots'},
    {'season': 2019, 'winning_team': 'Chiefs'},
    {'season': 2020, 'winning_team': 'Buccaneers'},
    {'season': 2021, 'winning_team': 'Rams'},
    {'season': 2022, 'winning_team': 'Chiefs'},
    {'season': 2023, 'winning_team': 'Chiefs'}
]
superbowl_winners_set = {(winner['season'], winner['winning_team']) for winner in superbowl_winners}

dftotal['is_superbowl_winner'] = dftotal.apply(
    lambda row: 1 if (row['season'], row['home']) in superbowl_winners_set else 0,
    axis=1
)

#print(dftotal.info())
#print(dftotal)

# ML MODEL STARTS HERE
dftotal_encoded = pd.get_dummies(dftotal, columns=['home', 'away'], drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

features = dftotal_encoded.drop(['is_superbowl_winner'], axis=1)
X = features  # Features for training
y = dftotal_encoded['is_superbowl_winner']  # Target variable for Super Bowl winner

# 80/20 train/test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)  # Increase iterations to avoid convergence warnings
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')