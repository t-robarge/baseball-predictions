import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import statsapi
import sys
import os
from PIL import Image, ImageTk
import io
import requests
from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# Import the original code classes
from no_wins_model import DataFetcher, DataProcessor, ModelTrainer


class MLBPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MLB Win Prediction Visualizer")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.bg_color = "#f0f0f0"
        self.header_color = "#162B4B"  # MLB blue
        self.accent_color = "#D50032"  # MLB red
        self.text_color = "#333333"
        
        self.root.configure(bg=self.bg_color)
        
        # Configure styles
        self.style.configure('Header.TLabel', background=self.header_color, foreground='white', font=('Arial', 16, 'bold'), padding=10)
        self.style.configure('Title.TLabel', background=self.bg_color, foreground=self.header_color, font=('Arial', 14, 'bold'), padding=5)
        self.style.configure('Data.TLabel', background=self.bg_color, foreground=self.text_color, font=('Arial', 11))
        self.style.configure('Accent.TButton', background=self.accent_color, foreground='white', font=('Arial', 12, 'bold'))
        self.style.map('Accent.TButton', background=[('active', '#B5002B')])
        
        # Initialize data and model classes
        self.data_fetcher = DataFetcher()
        self.model_trainer = ModelTrainer()
        self.division_results = {}
        self.overall_accuracy = 0.0
        self.rmse = 0.0
        
        # Setup main containers
        self.setup_header()
        self.setup_content()
        
        # Initialize with the current year
        current_year = 2024  # Default to 2024 for demo purposes
        self.year_var.set(str(current_year))
        
    def setup_header(self):
        header_frame = ttk.Frame(self.root, style='Header.TFrame')
        header_frame.pack(fill=tk.X)
        
        # Logo and title
        title_label = ttk.Label(header_frame, text="MLB Win Prediction Visualizer", style='Header.TLabel')
        title_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Try to fetch and display MLB logo
        try:
            logo_url = "https://www.mlbstatic.com/team-logos/league-on-dark/1.svg"
            response = requests.get(logo_url)
            if response.status_code == 200:
                img_data = response.content
                img = Image.open(io.BytesIO(img_data))
                img = img.resize((80, 40), Image.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(img)
                logo_label = ttk.Label(header_frame, image=self.logo_img, background=self.header_color)
                logo_label.pack(side=tk.RIGHT, padx=10, pady=5)
        except:
            # If logo fetch fails, display a text alternative
            logo_alt = ttk.Label(header_frame, text="MLB", style='Header.TLabel')
            logo_alt.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def setup_content(self):
        # Main content area
        content_frame = ttk.Frame(self.root, padding=10)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(content_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Year selection
        year_frame = ttk.Frame(control_frame)
        year_frame.pack(fill=tk.X, pady=10)
        
        year_label = ttk.Label(year_frame, text="Projection Year:", style='Data.TLabel')
        year_label.pack(side=tk.LEFT, padx=5)
        
        self.year_var = tk.StringVar()
        years = [str(year) for year in range(2013, 2025) if year != 2020]
        year_combo = ttk.Combobox(year_frame, textvariable=self.year_var, values=years, width=6)
        year_combo.pack(side=tk.LEFT, padx=5)
        
        # Model selection
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=10)
        
        model_label = ttk.Label(model_frame, text="Model:", style='Data.TLabel')
        model_label.pack(side=tk.LEFT, padx=5)
        
        self.model_var = tk.StringVar(value="BayesianRidge")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=["BayesianRidge", "LinearRegression", "RidgeCV"], width=15)
        model_combo.pack(side=tk.LEFT, padx=5)
        
        # Run button
        run_button = ttk.Button(control_frame, text="Run Projection", command=self.run_projection, style='Accent.TButton')
        run_button.pack(fill=tk.X, pady=10)
        
        # Information display
        info_frame = ttk.LabelFrame(control_frame, text="Model Information", padding=10)
        info_frame.pack(fill=tk.X, pady=10)
        
        self.rmse_var = tk.StringVar(value="RMSE: -")
        rmse_label = ttk.Label(info_frame, textvariable=self.rmse_var, style='Data.TLabel')
        rmse_label.pack(anchor=tk.W, pady=2)
        
        self.accuracy_var = tk.StringVar(value="Overall Accuracy: -")
        accuracy_label = ttk.Label(info_frame, textvariable=self.accuracy_var, style='Data.TLabel')
        accuracy_label.pack(anchor=tk.W, pady=2)
        
        # Help
        help_button = ttk.Button(control_frame, text="Help/About", command=self.show_help)
        help_button.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Right area for results
        self.results_frame = ttk.Notebook(content_frame)
        self.results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Placeholder tabs
        self.tables_tab = ttk.Frame(self.results_frame, padding=10)
        self.graphs_tab = ttk.Frame(self.results_frame, padding=10)
        self.heatmap_tab = ttk.Frame(self.results_frame, padding=10)
        
        self.results_frame.add(self.tables_tab, text="Division Tables")
        self.results_frame.add(self.graphs_tab, text="Bar Charts")
        self.results_frame.add(self.heatmap_tab, text="Prediction Heatmap")
        
        # Status bar
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding=2)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Ready. Select a projection year and click 'Run Projection'")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X)
    
    def run_projection(self):
        # Clear previous results
        for widget in self.tables_tab.winfo_children():
            widget.destroy()
        for widget in self.graphs_tab.winfo_children():
            widget.destroy()
        for widget in self.heatmap_tab.winfo_children():
            widget.destroy()
            
        self.status_var.set("Running projection... Please wait.")
        self.root.update()
        
        try:
            projection_year = int(self.year_var.get())
            selected_model = self.model_var.get()
            
            # Run model projection (based on the original code)
            self.run_model(projection_year, selected_model)
            
            # Update status
            self.status_var.set(f"Projection completed for {projection_year} using {selected_model}")
            self.rmse_var.set(f"RMSE: {self.rmse:.2f}")
            self.accuracy_var.set(f"Overall Accuracy: {self.overall_accuracy*100:.2f}%")
            
            # Display results
            self.display_division_tables()
            self.display_bar_charts()
            self.display_heatmap()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred during projection.")
    
    def run_model(self, projection_year, selected_model):
        # This follows the logic from the original Main.run() method
        season_list = [i for i in range(2013, 2025) if i != 2020 and i != projection_year]
        team_list = self.data_fetcher.team_ids.values()
        
        # Compile and export training data
        training_data = {season * 1000 + team: {} for team in team_list for season in season_list}
        for season in season_list:
            training_data = self.data_fetcher.compile_team_data(team_list, training_data, season)
        self.data_fetcher.export_to_csv(training_data, filename="mlb_stats_training.csv")

        # Compile and export test data
        test_data = {projection_year * 1000 + team: {} for team in team_list}
        test_data = self.data_fetcher.compile_team_data(team_list, test_data, projection_year)
        self.data_fetcher.export_to_csv(test_data, filename="mlb_stats_test.csv")
        
        # Preprocess training data
        data_processor = DataProcessor(filename="mlb_stats_training.csv")
        df_scaled, y_labels = data_processor.preprocess_data()
        X_train = df_scaled
        y_train = y_labels
        
        # Initialize the selected model
        if selected_model == "BayesianRidge":
            model = BayesianRidge()
        elif selected_model == "LinearRegression":
            model = LinearRegression()
        elif selected_model == "RidgeCV":
            model = RidgeCV()
        
        # Train the model
        self.model_trainer = ModelTrainer(model=model)
        self.model_trainer.train(X_train, y_train)
        
        # Preprocess test data
        data_processor = DataProcessor(filename="mlb_stats_test.csv")
        df_scaled_test, y_labels_test = data_processor.preprocess_data()
        
        # Make predictions and evaluate
        y_pred = self.model_trainer.predict(df_scaled_test)
        self.rmse = self.model_trainer.evaluate(y_labels_test, y_pred)
        
        # Get results data
        self.results_df, self.division_accuracies = self.prepare_results(df_scaled_test, y_labels_test, y_pred, projection_year)
        self.overall_accuracy = sum(acc for acc in self.division_accuracies.values()) / len(self.division_accuracies)
    
    def prepare_results(self, df_scaled_test, y_labels_test, y_pred, projection_year):
        # Create a mapping of team IDs to abbreviations
        team_id_to_abbreviation = {team_id: abbrev for abbrev, team_id in self.data_fetcher.team_ids.items()}
        
        # Replace team IDs with abbreviations in the results DataFrame
        results_df = pd.DataFrame({
            'Actual Wins': y_labels_test,
            'Predicted Wins': y_pred
        }, index=df_scaled_test.index)
        
        # Map team IDs to abbreviations
        results_df.index = results_df.index.map(lambda x: team_id_to_abbreviation[x % 1000])
        
        # Add the 'Difference' column
        results_df['Difference'] = results_df['Actual Wins'] - results_df['Predicted Wins']
        
        # Add division information and division ranks
        division_mapping, division_ranks = self.get_division_mapping(projection_year)
        results_df['Division'] = results_df.index.map(division_mapping)
        results_df['Division Rank'] = results_df.index.map(division_ranks)
        
        # Calculate ranking accuracy for predicted wins by division
        division_accuracies = {}
        for division, group in results_df.groupby('Division'):
            predicted_accuracy = self.calculate_ranking_accuracy(
                group['Predicted Wins'], 
                group['Actual Wins'], 
                group['Division Rank']
            )
            division_accuracies[division] = predicted_accuracy
        
        # Sort by division and predicted wins
        results_df = results_df.sort_values(by=['Division', 'Predicted Wins'], ascending=[True, False])
        
        return results_df, division_accuracies
    
    def calculate_ranking_accuracy(self, predicted, actual, division_ranks):
        # Convert predicted wins to ranks within each division
        predicted_ranks = predicted.rank(method='min', ascending=False)
        
        # Use the actual division ranks
        actual_ranks = pd.Series(division_ranks)
        
        # Calculate Spearman correlation coefficient
        correlation, _ = spearmanr(predicted_ranks, actual_ranks)
        return correlation
    
    def get_division_mapping(self, year):
        # This is taken directly from the original code
        standings = statsapi.get("standings", {'season': year, 'sportIds': 1, 'leagueId': "103,104"})
        division_mapping = {}
        division_ranks = {}
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
                division_ranks[team_abbrev] = team['divisionRank']

        return division_mapping, division_ranks

    def display_division_tables(self):
        # Create a canvas with scrollbar for the tables
        canvas_frame = ttk.Frame(self.tables_tab)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Group by division
        division_groups = self.results_df.groupby('Division')
        
        # Division colors (for visual distinction)
        division_colors = {
            "AL East": "#BA0C2F",   # Red
            "AL Central": "#0C2340", # Navy
            "AL West": "#003831",    # Dark Green
            "NL East": "#002D72",    # Blue
            "NL Central": "#31006F", # Purple
            "NL West": "#FD5A1E"     # Orange
        }
        
        # Display each division in a table
        for i, (division, group) in enumerate(division_groups):
            # Division header
            division_frame = ttk.Frame(scrollable_frame, padding=10)
            division_frame.pack(fill=tk.X, pady=10)
            
            header_bg = division_colors.get(division, self.header_color)
            division_label = ttk.Label(
                division_frame, 
                text=f"{division} (Accuracy: {self.division_accuracies[division]*100:.2f}%)",
                background=header_bg,
                foreground="white",
                font=('Arial', 12, 'bold'),
                padding=5
            )
            division_label.pack(fill=tk.X)
            
            # Table for this division
            table_frame = ttk.Frame(division_frame)
            table_frame.pack(fill=tk.X, pady=5)
            
            # Table headers
            headers = ["Team", "Actual Wins", "Predicted Wins", "Difference"]
            for j, header in enumerate(headers):
                header_label = ttk.Label(
                    table_frame, 
                    text=header, 
                    font=('Arial', 11, 'bold'),
                    background="#DDDDDD",
                    padding=5,
                    borderwidth=1,
                    relief="solid"
                )
                header_label.grid(row=0, column=j, sticky="nsew")
                table_frame.grid_columnconfigure(j, weight=1)
            
            # Table data
            for k, (idx, row) in enumerate(group.iterrows(), 1):
                # Team name
                team_label = ttk.Label(
                    table_frame, 
                    text=idx, 
                    padding=5,
                    borderwidth=1,
                    relief="solid"
                )
                team_label.grid(row=k, column=0, sticky="nsew")
                
                # Actual wins
                actual_label = ttk.Label(
                    table_frame, 
                    text=f"{row['Actual Wins']:.0f}", 
                    padding=5,
                    borderwidth=1,
                    relief="solid"
                )
                actual_label.grid(row=k, column=1, sticky="nsew")
                
                # Predicted wins
                pred_label = ttk.Label(
                    table_frame, 
                    text=f"{row['Predicted Wins']:.1f}", 
                    padding=5,
                    borderwidth=1,
                    relief="solid"
                )
                pred_label.grid(row=k, column=2, sticky="nsew")
                
                # Difference with color coding
                diff = row['Difference']
                if diff > 0:
                    diff_color = "#E6FFE6"  # Light green (better than predicted)
                elif diff < 0:
                    diff_color = "#FFE6E6"  # Light red (worse than predicted)
                else:
                    diff_color = "#FFFFFF"  # White (exactly as predicted)
                    
                diff_label = ttk.Label(
                    table_frame, 
                    text=f"{diff:.1f}", 
                    padding=5,
                    borderwidth=1,
                    relief="solid",
                    background=diff_color
                )
                diff_label.grid(row=k, column=3, sticky="nsew")
    
    def display_bar_charts(self):
        # Create tabs for different divisions in the graphs tab
        graph_notebook = ttk.Notebook(self.graphs_tab)
        graph_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create a tab for each division
        division_groups = self.results_df.groupby('Division')
        
        for division, group in division_groups:
            # Create a frame for this division
            division_frame = ttk.Frame(graph_notebook)
            graph_notebook.add(division_frame, text=division)
            
            # Create the bar chart
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Sort teams by actual wins for better visualization
            group = group.sort_values('Actual Wins', ascending=False)
            
            # Set width and positions for bars
            width = 0.35
            ind = np.arange(len(group))
            
            # Create bars
            actual_bars = ax.bar(ind - width/2, group['Actual Wins'], width, label='Actual Wins', color='#1E88E5')
            predicted_bars = ax.bar(ind + width/2, group['Predicted Wins'], width, label='Predicted Wins', color='#FFC107')
            
            # Add labels and title
            ax.set_xlabel('Teams')
            ax.set_ylabel('Wins')
            ax.set_title(f'{division} - Actual vs Predicted Wins')
            ax.set_xticks(ind)
            ax.set_xticklabels(group.index, rotation=45, ha='right')
            ax.legend()
            
            # Add values above bars
            for bar in actual_bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
                
            for bar in predicted_bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            # Add accuracy information
            accuracy = self.division_accuracies[division]
            ax.annotate(f'Ranking Accuracy: {accuracy*100:.2f}%',
                        xy=(0.5, 0.02),
                        xycoords='figure fraction',
                        ha='center',
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
            
            fig.tight_layout()
            
            # Add the figure to the frame
            canvas = FigureCanvasTkAgg(fig, master=division_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_heatmap(self):
        # Create a heatmap showing the differences between actual and predicted wins
        fig = plt.figure(figsize=(12, 8))
        
        # Create a custom arrangement of subplots
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        division_order = [
            "AL East", "NL East",
            "AL Central", "NL Central",
            "AL West", "NL West"
        ]
        
        # Define a diverging color scheme (red to blue)
        cmap = plt.cm.RdBu_r  # Red for overperformance, Blue for underperformance
        norm = plt.Normalize(-20, 20)  # Adjust range as needed
        
        # Create subplots for each division
        for i, division in enumerate(division_order):
            ax = fig.add_subplot(gs[i//2, i%2])
            
            # Get the data for this division
            try:
                div_data = self.results_df[self.results_df['Division'] == division]
                
                # Sort by actual wins
                div_data = div_data.sort_values('Actual Wins', ascending=False)
                
                # Extract the data for plotting
                teams = div_data.index
                actual = div_data['Actual Wins']
                predicted = div_data['Predicted Wins']
                diff = div_data['Difference']
                
                # Create horizontal bar chart
                bars = ax.barh(teams, predicted, color=cmap(norm(diff)), alpha=0.7, label='Predicted')
                ax.barh(teams, actual, color='none', edgecolor='black', linestyle='--', 
                        linewidth=2, label='Actual')
                
                # Add labels for predicted wins
                for j, (team, pred, act) in enumerate(zip(teams, predicted, actual)):
                    # Determine the furthest point (max between predicted and actual)
                    furthest_point = max(pred, act)
                    
                    # Place the label after the furthest point
                    ax.text(furthest_point + 1, j, f'{pred:.1f}', 
                            va='center', ha='left', fontsize=8, color='black')
                
                # Set title and labels
                ax.set_title(f'{division} (Acc: {self.division_accuracies[division]*100:.1f}%)')
                ax.set_xlabel('Wins')
                ax.set_xlim(0, max(actual.max(), predicted.max()) * 1.1)
                
                # Add a grid for better readability
                ax.grid(axis='x', linestyle='--', alpha=0.6)
                
                # Legend (only for the first subplot)
                if i == 0:
                    ax.legend(loc='upper right', fontsize=8)
                    
            except Exception as e:
                # Handle any errors (like division not found)
                ax.text(0.5, 0.5, f"Error displaying {division}: {str(e)}", 
                    ha='center', va='center', transform=ax.transAxes)
        
        # Add a color bar to explain the color scale
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                        cax=cbar_ax)
        cbar.set_label('Actual - Predicted Wins\n(Red = Overperformed, Blue = Underperformed)')
        
        # Add overall title
        fig.suptitle(f'MLB Win Predictions - Year {self.year_var.get()}\nOverall Accuracy: {self.overall_accuracy*100:.2f}%, RMSE: {self.rmse:.2f}', 
                    fontsize=16)
        
        fig.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # Add the figure to the frame
        canvas = FigureCanvasTkAgg(fig, master=self.heatmap_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_help(self):
        help_text = """MLB Win Prediction Visualizer

    This application visualizes predictions from different regression models for MLB team wins.

    How to use:
    1. Select a projection year from the dropdown
    2. Select a model from the dropdown (BayesianRidge, LinearRegression, or RidgeCV)
    3. Click "Run Projection" to process the data
    4. View results in different tabs:
    - Division Tables: Detailed tables showing actual vs. predicted wins
    - Bar Charts: Visual comparison of actual vs. predicted wins by division
    - Prediction Heatmap: Overview of all divisions with color-coded performance indicators

    Model information:
    - BayesianRidge: A Bayesian approach to linear regression
    - LinearRegression: Ordinary least squares linear regression
    - RidgeCV: Ridge regression with built-in cross-validation

    The accuracy is measured using Spearman rank correlation within divisions
    RMSE (Root Mean Square Error) shows the overall prediction accuracy in terms of wins

    Data is sourced from the MLB StatsAPI.
    """
        messagebox.showinfo("Help & Information", help_text)
def main():
    root = tk.Tk()
    app = MLBPredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()