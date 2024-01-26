import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import expit
from scipy.stats import entropy

class Team:
    def __init__(self, name, elo):
        self.name = name
        self.elo = elo
    
    def __repr__(self):
        return f'{self.name}: {self.elo}'
    

class BradleyTerryGame:
    def __init__(self, team1, team2):
        self.team1 = team1
        self.team2 = team2

    def simulate(self):
        skill_diff = self.team1.elo - self.team2.elo
        prob_team1 = expit(skill_diff)
        prob_team2 = 1 - prob_team1

        result = np.random.choice([self.team1.name, self.team2.name], p=[prob_team1, prob_team2])
        if result == self.team1.name:
            self.team1.elo, self.team2.elo = self.elo_update(self.team1.elo, self.team2.elo)
        else:
            self.team2.elo, self.team1.elo = self.elo_update(self.team2.elo, self.team1.elo)

    def elo_update(self, winner_elo, loser_elo):
        elo_diff = winner_elo - loser_elo
        if elo_diff < 0:
            mult = 1
        elif elo_diff == 0:
            mult = 1
        else:
            mult = 1 + ((elo_diff) / 1000)
        new_winner_elo = winner_elo + 32 * mult
        new_loser_elo = loser_elo - 32 * mult
        return new_winner_elo, new_loser_elo
    
    def __repr__(self):
        return f'{self.team1.name} vs {self.team2.name}'
    
    def __str__(self):
        return f'{self.team1.name} vs {self.team2.name}'
    
    def __eq__(self, other):
        return self.team1 == other.team1 and self.team2 == other.team2
        
def select_next_matches(teams, n_matches):
    """
    Selects the next matches that maximize ranking information gain.
    """
    potential_games = []
    for i, team1 in enumerate(teams):
        for team2 in teams[i+1:]:
            skill_diff = abs(team1.elo - team2.elo)
            potential_games.append((skill_diff, BradleyTerryGame(team1, team2)))

    # Sort games by skill difference (lower difference has higher information gain)
    potential_games.sort(key=lambda x: x[0])

    # Select the top N matches
    return [game for _, game in potential_games[:n_matches]]

class Season:
    def __init__(self, teams):
        self.teams = teams
        self.games = []

    def current_rankings(self):
        """
        Return current rankings of the teams.
        """
        return sorted(self.teams, key=lambda x: x.elo, reverse=True)

    def ranking_uncertainty(self):
        """
        Calculate and return the entropy-based uncertainty of the current rankings.
        """
        elos = np.array([team.elo for team in self.teams])
        # Normalize ELO scores to prevent overflow
        elos -= np.max(elos)
        elo_probs = np.exp(elos) / np.sum(np.exp(elos))
        return entropy(elo_probs)
    

    def generate_next_season(self, n_matches):
        """
        Generate the game for each team that maximizes ranking information gain.
        """
        next_season = Season(self.teams)
        next_season.games = self.games.copy()
        next_season.games += select_next_matches(self.teams, n_matches)
        return next_season

    def add_game_result(self, game):
        """
        Update the Season with the result of a game.
        """
        self.games.append(game)
        game.simulate()  # Simulate the game to update team Elo ratings

    def __str__(self):
        return '\n'.join([str(team) for team in self.current_rankings()])

# Example of how to use the Season class
teams = []
df = pd.read_csv('data/testing_data.csv')
for i, row in df.iterrows():
    teams.append(Team(row['proj'], round(1200*0.01*row['score'], 2)))
season = Season(teams)
print("Current Rankings:\n", season)
print("Uncertainty:", season.ranking_uncertainty())

