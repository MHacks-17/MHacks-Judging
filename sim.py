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
        expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        expected_loser = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))

        # Sensitivity 
        K = 20
        new_winner_elo = winner_elo + K * (1 - expected_winner)
        new_loser_elo = loser_elo + K * (0 - expected_loser)

        return new_winner_elo, new_loser_elo
    
    def __repr__(self):
        return f'{self.team1.name} vs {self.team2.name}'
    
    def __str__(self):
        return f'{self.team1.name} vs {self.team2.name}'
    
    def __eq__(self, other):
        return self.team1 == other.team1 and self.team2 == other.team2
        
teams = []
for i in range(10):
    teams.append(Team("Team " + str(i), 1200))

class Rankings():
    def __init__(self, teams):
        self.teams = teams
        self.rankings = pd.DataFrame(columns=['Team', 'Elo'])
        self.rankings['Team'] = [team.name for team in self.teams]
        self.rankings['Elo'] = [team.elo for team in self.teams]
        self.rankings = self.rankings.sort_values(by='Elo', ascending=False).reset_index(drop=True)

    def update(self):
        self.rankings['Elo'] = [team.elo for team in self.teams]
        self.rankings = self.rankings.sort_values(by='Elo', ascending=False).reset_index(drop=True)

    def entropy(self):
        return entropy(self.rankings['Elo'])        
