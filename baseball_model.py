import pandas as pd
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsapi
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

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
        stats_needed = [
            ('runs', 'pitching'),
            ('era', 'pitching'),
            ('whip', 'pitching'),
            ('wins', 'pitching'),
            ('strikeoutWalkRatio', 'pitching'),
            ('ops', 'pitching'),
            ('ops', 'hitting'),
            ('leftOnBase', 'hitting'),
            ('runs', 'hitting'),
            ('hits', 'hitting'),
            ('avg', 'hitting'),
            # ('earnedRuns', 'pitching'),
            # ('slg', 'hitting'),
            # ('obp', 'hitting'),
            # ('stolenBases', 'hitting'),
            # ('blownSaves', 'pitching'),
            # ('groundOuts', 'hitting'),
            # ('airOuts', 'hitting'),
            # ('doubles', 'hitting'),
            # ('triples', 'hitting'),
            # ('homeRuns', 'hitting'),
            # ('strikeOuts', 'hitting'),
            # ('baseOnBalls', 'hitting'),
            # ('intentionalWalks', 'hitting'),
            # ('hitByPitch', 'hitting'),
            # ('atBats', 'hitting'),
            # ('caughtStealing', 'hitting'),
            # ('stolenBasePercentage', 'hitting'),
            # ('groundIntoDoublePlay', 'hitting'),
            # ('numberOfPitches', 'hitting'),
            # ('plateAppearances', 'hitting'),
            # ('totalBases', 'hitting'),
            # ('rbi', 'hitting'),
            # ('sacBunts', 'hitting'),
            # ('sacFlies', 'hitting'),
            # ('groundOuts', 'pitching'),
            # ('airOuts', 'pitching'),
            # ('doubles', 'pitching'),
            # ('triples', 'pitching'),
            # ('strikeOuts', 'pitching'),
            # ('hitByPitch', 'pitching'),
            # ('avg', 'pitching'),
            # ('obp', 'pitching'),
            # ('slg', 'pitching'),
            # ('atBats', 'pitching'),
            # ('caughtStealing', 'pitching'),
            # ('stolenBases', 'pitching'),
            # ('stolenBasePercentage', 'pitching'),
            # ('groundIntoDoublePlay', 'pitching'),
            # ('saveOpportunities', 'pitching'),
            # ('holds', 'pitching'),
            # ('battersFaced', 'pitching'),
            # ('outs', 'pitching'),
            # ('shutouts', 'pitching'),
            # ('strikes', 'pitching'),
            # ('strikePercentage', 'pitching'),
            # ('hitBatsmen', 'pitching'),
            # ('balks', 'pitching'),
            # ('wildPitches', 'pitching'),
            # ('pickoffs', 'pitching'),
            # ('totalBases', 'pitching'),
            # ('groundOutsToAirouts', 'pitching'),
            # ('pitchesPerInning', 'pitching'),
            # ('strikeoutsPer9Inn', 'pitching'),
            # ('walksPer9Inn', 'pitching'),
            # ('hitsPer9Inn', 'pitching'),
            # ('runsScoredPer9', 'pitching'),
            # ('homeRunsPer9', 'pitching'),
        ]
        gp_dict = dict()
        gp = statsapi.get("teams_stats", {'stats': 'byDateRange',
                                             'season': season,
                                             'group': "hitting",
                                             'gameType': 'R',
                                             'startDate': f"03/01/{season}",
                                             'endDate': f"07/07/{season}",
                                             'sportIds': 1}, )
        gp = gp['stats'][0]['splits']
        for team in gp:
            gp_dict[team['team']['id']] = team['stat']['gamesPlayed']
        for my_stat in stats_needed:
            dic = self.get_stat(stat=my_stat[0], group=my_stat[1], season=season)
            for team_id in team_list:
                if my_stat[0] in ['avg', 'obp', 'slg', 'ops', 'era', 'whip', 'strikeoutWalkRatio', 'stolenBasePercentage', 'groundOutsToAirouts', 'pitchesPerInning', 'strikeoutsPer9Inn', 'walksPer9Inn', 'hitsPer9Inn', 'runsScoredPer9', 'homeRunsPer9', 'strikePercentage', ]:
                    if my_stat[1] == 'pitching':
                        my_dict[season * 1000 + team_id]['p_' + my_stat[0]] = dic[team_id]
                    else: 
                        my_dict[season * 1000 + team_id]['h_' + my_stat[0]] = dic[team_id]
                else:
                    if my_stat[1] == 'pitching':
                        my_dict[season * 1000 + team_id]['p_' + my_stat[0]] = dic[team_id]/gp_dict[team_id]
                    else: 
                        my_dict[season * 1000 + team_id]['h_' + my_stat[0]] = dic[team_id]/gp_dict[team_id]
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
        mid_season_wins = self.df['p_wins'].copy()
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(self.df)
        df_scaled = pd.DataFrame(scaled_array, columns=self.df.columns, index=self.df.index)
        return df_scaled, y_labels, mid_season_wins

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
    def compare_models(self,X_train,X_test,y_train,y_test):
        "Compares 3 linear regression models on the set of test and training data provided"
        br = BayesianRidge()
        lr = LinearRegression()
        rr = Ridge()
        rmses = []
        mses = []
        for model in [br,lr,rr]:
            print(model)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            print(y_pred)
            mse = mean_squared_error(y_test,y_pred)
            rmse = mse**(1/2)
            mses.append(mse)
            rmses.append(rmse)  
        results_rmse = {
            "Model": ["BayesianRidge", "LinearRegression", "Ridge"],
            "Metric": rmse
        }
        results_mse = {
            "Model": ["BayesianRidge", "LinearRegression", "Ridge"],
            "Metric": mse
        }
        # rmse
        df_results = pd.DataFrame(results_rmse)
        plt.figure(figsize=(8, 5))
        bars = plt.bar(df_results["Model"], df_results["Metric"], color="skyblue")
        plt.xlabel("Model")
        plt.ylabel("RMSE")  
        plt.title("RMSE Comparison of Regression Models")
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        plt.savefig("RMSEComparison.png")
        # mse
        df_results = pd.DataFrame(results_mse)
        plt.figure(figsize=(8, 5))
        bars = plt.bar(df_results["Model"], df_results["Metric"], color="skyblue")
        plt.xlabel("Model")
        plt.ylabel("MSE")  
        plt.title("MSE Comparison of Regression Models")
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        plt.savefig("MSEComparison.png")

class Main:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.midseason_accuracy = None
        self.prediction_accuracy = None
        self.test = False
        self.display_results = True

        # Set pandas display options to prevent truncation
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.expand_frame_repr', False)

    def run(self, season = None, test = False,display_results=True):
        if not display_results: self.display_results = False

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
        X_train = df_scaled
        y_train = y_labels
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
        if self.print:
            self.display_standings(df_scaled_test, y_labels_test, y_pred, mid_season_wins_test, projection_year)
    def run_for_comparison(self,season=None):
        "creates bar graph comparison of models"
        # get list of teams
        if season:
            projection_year = season
            self.test = True
        else:
            projection_year = self.get_projection_year()
        season_list = self.get_season_list(projection_year)
        team_list = self.data_fetcher.team_ids.values()
        # split data based on test year
        self.compile_and_export_data(team_list, season_list, projection_year)
        # Preprocess training data
        self.data_processor = DataProcessor(filename="mlb_stats_training.csv")
        df_scaled, y_labels, mid_season_wins = self.data_processor.preprocess_data()
        X_train = df_scaled
        y_train = y_labels
        
        # Preprocess test data
        self.data_processor = DataProcessor(filename="mlb_stats_test.csv")
        df_scaled_test, y_labels_test, mid_season_wins_test = self.data_processor.preprocess_data()

        # Make predictions and evaluate the model
        self.model_trainer.compare_models(X_train,df_scaled_test,y_train,y_labels_test)
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
    # main.run()
    main.run_for_comparison(season=2014)
'''
    ##### For testing full range #####
    mid_acc = []
    pred_acc = []
    for year in range(2008,2025):
        if year != 2020:
            print(f"Running for projection year: {year}")
            main.run(season=year, test=True)
            mid_acc.append(main.midseason_accuracy)
            pred_acc.append(main.prediction_accuracy)
    print(f"Overall Midseason Accuracy: {sum(mid_acc)/len(mid_acc)}")
    print(f"Overall Prediction Accuracy: {sum(pred_acc)/len(pred_acc)}")
'''