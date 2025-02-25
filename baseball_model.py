import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, BayesianRidge, LinearRegression, Ridge, PoissonRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
import statsapi

class DataFetcher:
    def __init__(self):
        self.team_ids = self.get_team_ids()

    def get_team_ids(self):
        teams = statsapi.get('teams', {'season': 2021, 'sportId': 1})
        return {team['abbreviation']: team['id'] for team in teams['teams']}

    def get_stat(self, group='pitching', stat='era', season=0):
        stat_dic = {}
        stats = statsapi.get("teams_stats", {'stats': 'byDateRange',
                                             'season': season,
                                             'group': group,
                                             'gameType': 'R',
                                             'startDate': f"03/01/{season}",
                                             'endDate': f"07/07/{season}",
                                             'sportIds': 1}, )
        splits = stats['stats'][0].get('splits', [])
        for split in splits:
            stat_dic[split['team']['id']] = split['stat'][stat]
        return stat_dic

    def compile_team_data(self, team_list, my_dict, season):
        stats_needed = [('era', 'pitching'), ('avg', 'hitting'), ('wins', 'pitching'), ('obp', 'hitting'),
                        ('slg', 'hitting'), ('ops', 'hitting'), ('stolenBases', 'hitting'), ('leftOnBase', 'hitting'),
                        ('runs', 'hitting'), ('whip', 'pitching'), ('runs', 'pitching'), ('blownSaves', 'pitching'),
                        ('strikeoutWalkRatio', 'pitching')]
        for my_stat in stats_needed:
            dic = self.get_stat(stat=my_stat[0], group=my_stat[1], season=season)
            for team_id in team_list:
                if my_stat[0] == 'wins':
                    my_dict[season * 1000 + team_id]['cur_wins'] = dic[team_id]
                elif my_stat[0] == 'runs' and my_stat[1] == 'pitching':
                    my_dict[season * 1000 + team_id]['runs_against'] = dic[team_id]
                else:
                    my_dict[season * 1000 + team_id][my_stat[0]] = dic[team_id]
        stats = statsapi.get("standings", {'season': season,
                                           'sportIds': 1,
                                           'leagueId': "103,104"})
        for division in stats['records']:
            for team in division['teamRecords']:
                my_dict[season * 1000 + team['team']['id']]['wins'] = team['wins']
        return my_dict

    def export_to_csv(self, data, filename="mlb_stats.csv"):
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.name = "team_id"
        df.to_csv(filename, index=True)

class DataProcessor:
    def __init__(self, filename="mlb_stats.csv"):
        self.df = pd.read_csv(filename)
        self.df.set_index(self.df.columns[0], inplace=True)

    def preprocess_data(self):
        y_labels = self.df.pop('wins')
        mid_season_wins = self.df['cur_wins'].copy()
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(self.df)
        df_scaled = pd.DataFrame(scaled_array, columns=self.df.columns, index=self.df.index)
        return df_scaled, y_labels, mid_season_wins

    def split_data(self, df_scaled, y_labels, test_size=0.2, random_state=42):
        return train_test_split(df_scaled, y_labels, test_size=test_size, random_state=random_state)


class ModelTrainer:
    def __init__(self):
        self.model = BayesianRidge()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** (1 / 2)
        return rmse


class Main:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()

    def run(self):
        # Fetch and compile data
        season_list = [i for i in range(2008, 2025) if i != 2020]
        team_list = self.data_fetcher.team_ids.values()
        my_dict = {season * 1000 + team: {} for team in team_list for season in season_list}
        for season in season_list:
            my_dict = self.data_fetcher.compile_team_data(team_list, my_dict, season)
        self.data_fetcher.export_to_csv(my_dict)
        
        # Reinitialize to ensure CSV updates
        self.data_processor = DataProcessor()

        # Preprocess data
        df_scaled, y_labels, mid_season_wins = self.data_processor.preprocess_data()
        X_train, X_test, y_train, y_test = self.data_processor.split_data(df_scaled, y_labels)

        # Train model and make predictions
        self.model_trainer.train(X_train, y_train)
        y_pred = self.model_trainer.predict(X_test)

        # Evaluate model
        rmse = self.model_trainer.evaluate(y_test, y_pred)
        print("RMSE:", rmse)


if __name__ == "__main__":
    main = Main()
    main.run()