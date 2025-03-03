# baseball-predictions
Probabalistic Modeling for Baseball

# Result Validation
Midseason wins are a very strong predictor for the end of season rankings, so we made sure to verify that there was an improvement when incorporating additional statistics. We evaluated this accuracy based off the Spearman Correlation Coefficient.

For seasons 2008-2024 (exluding 2020 due to COVID) we found the following:
    Overall Midseason Accuracy: 0.8584506628856584
    Overall Prediction Accuracy: 0.8703498163546692

Without considering midseason wins: 
    Overall Prediction Accuracy: 0.82015907332374

### Credits
This project uses the mlbstatsapi library by toddrob, which is licensed under the GNU General Public License (GPL) Version 3. 
You can find the original library at: https://github.com/toddrob99/MLB-StatsAPI