import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsapi
from scipy.stats import spearmanr

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
        stats_needed = [('era', 'pitching'), ('wins', 'pitching'),
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
        self.midseason_accuracy = None
        self.prediction_accuracy = None
        self.test = False

        # Set pandas display options to prevent truncation
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.expand_frame_repr', False)

    def run(self, season = None, test = False, ):
        if test:
            projection_year = season
            self.test = True
        else:
            projection_year = self.get_projection_year()
        season_list = self.get_season_list(projection_year)
        team_list = self.data_fetcher.team_ids.values()

        # Fetch and compile training and test data
        self.compile_and_export_data(team_list, season_list, projection_year)

        # Preprocess, train, and evaluate the model
        self.train_and_evaluate_model(projection_year)

        return 

    def get_projection_year(self):
        return int(input("Enter the year you would like to project: "))

    def get_season_list(self, projection_year):
        return [i for i in range(2013, 2025) if i != 2020 and i != projection_year]

    def compile_and_export_data(self, team_list, season_list, projection_year):
        # Compile and export training data
        training_data = {season * 1000 + team: {} for team in team_list for season in season_list}
        for season in season_list:
            training_data = self.data_fetcher.compile_team_data(team_list, training_data, season)
        self.data_fetcher.export_to_csv(training_data, filename="mlb_stats_training.csv")

        # Compile and export test data
        test_data = {projection_year * 1000 + team: {} for team in team_list}
        test_data = self.data_fetcher.compile_team_data(team_list, test_data, projection_year)
        self.data_fetcher.export_to_csv(test_data, filename="mlb_stats_test.csv")

    def train_and_evaluate_model(self, projection_year):
        # Preprocess training data
        self.data_processor = DataProcessor(filename="mlb_stats_training.csv")
        df_scaled, y_labels, mid_season_wins = self.data_processor.preprocess_data()
        X_train, X_test, y_train, y_test = self.data_processor.split_data(df_scaled, y_labels)

        # Train the model
        self.model_trainer.train(X_train, y_train)

        # Preprocess test data
        self.data_processor = DataProcessor(filename="mlb_stats_test.csv")
        df_scaled_test, y_labels_test, mid_season_wins_test = self.data_processor.preprocess_data()

        # Make predictions and evaluate the model
        y_pred = self.model_trainer.predict(df_scaled_test)
        rmse = self.model_trainer.evaluate(y_labels_test, y_pred)
        if not self.test: print(f"RMSE for {projection_year}:", rmse)

        # Display results
        self.display_standings(df_scaled_test, y_labels_test, y_pred, mid_season_wins_test, projection_year)

    def display_standings(self, df_scaled_test, y_labels_test, y_pred, mid_season_wins_test, projection_year):
        # Create a mapping of team IDs to abbreviations
        team_id_to_abbreviation = {team_id: abbrev for abbrev, team_id in self.data_fetcher.team_ids.items()}

        # Replace team IDs with abbreviations in the results DataFrame
        results_df = pd.DataFrame({
            'Mid-Season Wins': mid_season_wins_test.loc[df_scaled_test.index],
            'Actual Wins': y_labels_test,
            'Predicted Wins': y_pred
        }, index=df_scaled_test.index)

        # Map team IDs to abbreviations
        results_df.index = results_df.index.map(lambda x: team_id_to_abbreviation[x % 1000])

        # Add the 'Difference' column
        results_df['Difference'] = results_df['Actual Wins'] - results_df['Predicted Wins']

        # Add division information
        division_mapping = self.get_division_mapping(projection_year)
        results_df['Division'] = results_df.index.map(division_mapping)

        # Calculate ranking accuracy for mid-season wins and predicted wins
        mid_season_accuracy = self.calculate_ranking_accuracy(results_df['Mid-Season Wins'], results_df['Actual Wins'])
        predicted_accuracy = self.calculate_ranking_accuracy(results_df['Predicted Wins'], results_df['Actual Wins'])
        self.midseason_accuracy = mid_season_accuracy
        self.prediction_accuracy = predicted_accuracy

        # Sort and display results
        results_df = results_df.sort_values(by=['Division', 'Predicted Wins'], ascending=[True, False])
        if not self.test:
            print("\nResults for the Projection Year (Grouped by Division, Ordered by Predicted Wins):")
            for division, group in results_df.groupby('Division'):
                print(f"\n{division}")
                print(group.drop(columns=['Division']))
            
        # Create a new table with division name and both metrics
        if not self.test:
            print(f"\nMid-Season Wins Accuracy: {mid_season_accuracy*100:.2f}%")
            print(f"Predicted Wins Accuracy: {predicted_accuracy*100:.2f}%")

    def calculate_ranking_accuracy(self, predicted, actual):
        # Calculate Spearman correlation coefficient
        correlation, _ = spearmanr(predicted, actual)
        return correlation

    def get_division_mapping(self, year):
        standings = statsapi.get("standings", {'season': year, 'sportIds': 1, 'leagueId': "103,104"})
        division_mapping = {}
        team_id_to_abbreviation = {team_id: abbrev for abbrev, team_id in self.data_fetcher.team_ids.items()}

        for division in standings['records']:
            division_id = division['division']['id']

            if division_id == 201:
                division_name = "AL East"
            elif division_id == 202:
                division_name = "AL Central"
            elif division_id == 200:
                division_name = "AL West"
            elif division_id == 204:
                division_name = "NL East"
            elif division_id == 205:
                division_name = "NL Central"
            elif division_id == 203:
                division_name = "NL West"

            for team in division['teamRecords']:
                team_abbrev = team_id_to_abbreviation[team['team']['id']]
                division_mapping[team_abbrev] = division_name

        return division_mapping


if __name__ == "__main__":
    main = Main()
    main.run()

    ##### For testing full range #####
    # mid_acc = []
    # pred_acc = []
    # for year in range(2008,2025):
    #     if year != 2020:
    #         print(f"Running for projection year: {year}")
    #         main.run(season=year, test=True)
    #         mid_acc.append(main.midseason_accuracy)
    #         pred_acc.append(main.prediction_accuracy)
    # print(f"Overall Midseason Accuracy: {sum(mid_acc)/len(mid_acc)}")
    # print(f"Overall Prediction Accuracy: {sum(pred_acc)/len(pred_acc)}")