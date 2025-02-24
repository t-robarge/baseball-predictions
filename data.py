import statsapi
import pandas as pd

ver = "1.8.1"
#season = "2021"
# League IDs: AL - 103, NL - 104, for both, put in string: "103,104"
'''
All available stat parameter types:

[{'displayName': 'projected'}, 
{'displayName': 'projectedRos'}, 
{'displayName': 'yearByYear'}, 
{'displayName': 'yearByYearAdvanced'}, 
{'displayName': 'yearByYearPlayoffs'}, 
{'displayName': 'season'}, 
{'displayName': 'standard'}, 
{'displayName': 'advanced'}, 
{'displayName': 'career'}, 
{'displayName': 'careerRegularSeason'}, 
{'displayName': 'careerAdvanced'}, 
{'displayName': 'seasonAdvanced'}, 
{'displayName': 'careerStatSplits'}, 
{'displayName': 'careerPlayoffs'}, 
{'displayName': 'gameLog'}, 
{'displayName': 'playLog'}, 
{'displayName': 'pitchLog'}, 
{'displayName': 'metricLog'}, 
{'displayName': 'metricAverages'}, 
{'displayName': 'pitchArsenal'}, 
{'displayName': 'outsAboveAverage'}, 
{'displayName': 'expectedStatistics'}, 
{'displayName': 'sabermetrics'}, 
{'displayName': 'sprayChart'}, 
{'displayName': 'tracking'}, 
{'displayName': 'vsPlayer'}, 
{'displayName': 'vsPlayerTotal'}, 
{'displayName': 'vsPlayer5Y'}, 
{'displayName': 'vsTeam'}, 
{'displayName': 'vsTeam5Y'}, 
{'displayName': 'vsTeamTotal'}, 
{'displayName': 'lastXGames'}, 
{'displayName': 'byDateRange'}, 
{'displayName': 'byDateRangeAdvanced'}, 
{'displayName': 'byMonth'}, 
{'displayName': 'byMonthPlayoffs'}, 
{'displayName': 'byDayOfWeek'}, 
{'displayName': 'byDayOfWeekPlayoffs'}, 
{'displayName': 'homeAndAway'}, 
{'displayName': 'homeAndAwayPlayoffs'}, 
{'displayName': 'winLoss'}, 
{'displayName': 'winLossPlayoffs'}, 
{'displayName': 'rankings'}, 
{'displayName': 'rankingsByYear'}, 
{'displayName': 'statsSingleSeason'}, 
{'displayName': 'statsSingleSeasonAdvanced'}, 
{'displayName': 'hotColdZones'}, 
{'displayName': 'availableStats'}, 
{'displayName': 'opponentsFaced'}, 
{'displayName': 'gameTypeStats'}, 
{'displayName': 'firstYearStats'}, 
{'displayName': 'lastYearStats'}, 
{'displayName': 'statSplits'}, 
{'displayName': 'statSplitsAdvanced'}, 
{'displayName': 'atGameStart'}, 
{'displayName': 'vsOpponents'}, 
{'displayName': 'sabermetricsMultiTeam'}]
'''

def getTeamIds():
    teams = statsapi.get('teams', {'season': 2021, 'sportId': 1})
    return {team['abbreviation']: team['id'] for team in teams['teams']}

def getGamePks(team_ids):
    team_gamePks = dict()

    for id in team_ids.values():
        # Set start date so that it is always before opening day, end date always around All Star Break
        first_half_schedule = statsapi.schedule(team = id, start_date=f"03/01/{season}", end_date=f"07/15/{season}", leagueId="103,104")

        gamePks = []
        for game in first_half_schedule:
            # Ensure game is a regular game and is complete
            if game['game_type'] == 'R' and game['status'] == 'Final':
                gamePks.append(game["game_id"])
        
        team_gamePks[id] = gamePks
    
    return team_gamePks

# Endpoint: teams_stats
#   - Supports date ranges
#   - Is not returning MLB teams atm
'''
def getTeamStats(team_id, start = f"03/01/{season}", end = f"07/15/{season}",group='hitting'):
    stats = statsapi.get("teams_stats", {'stats' : 'byDateRange', 
                                         'season' : season, 
                                         'group' : group, 
                                         'gameType' : 'R', 
                                         'startDate' : start, 
                                         'endDate' : end,
                                         'sportIds':1,
                                         })
    splits = stats['stats'][0].get('splits', [])
    for split in splits:
        if split.get('team', {}).get('id') == team_id:
            return split.get('stat')
    return None
'''
# Endpoint: team_stats
#   - Doesn't seem to support date ranges (tried to use force = True with start and end date)
#   - WORKS FOR FULL SEASON if we want to try that (change 'group' parameter for different stat types)
'''
def getTeamStats(team_id, start = f"03/01/{season}", end = f"07/15/{season}"):
    stats = statsapi.get("team_stats", {'teamId' : team_id, 
                                        'stats' : 'byDateRange', # Ineffective without working start/end params?
                                        'season' : season, 
                                        'group' : 'hitting', 
                                        'gameType' : 'R', 
                                        'startDate' : start, # Not a param
                                        'endDate' : end, # Not a param
                                        'force' : True # Can have unexpected results
                                        })
    return stats
'''

# Get abbreviation : id team dictionary
# team_ids = getTeamIds()

# Can get boxscore data iteratively from game ids if necessary, would take a while tho due to request response time
# gamePks = getGamePks(team_ids)

# Testing
#print(getTeamStats(133,group='pitching'))

gameTypes = statsapi.get("meta",{
                     "type":"statGroups"})
#print(gameTypes)
## variation in games played. If needed, fine tune date range to limit variation
def getStat(group='pitching',stat='era',season=0):
    stat_dic = {}
    stats = statsapi.get("teams_stats", {'stats' : 'byDateRange', 
                                         'season' : season, 
                                         'group' : group, 
                                         'gameType' : 'R', 
                                         'startDate' : f"03/01/{season}", 
                                         'endDate' : f"07/07/{season}",
                                         'sportIds':1},)
    splits = stats['stats'][0].get('splits', [])    
    for split in splits:
        stat_dic[split['team']['id']] = split['stat'][stat]
    return stat_dic

def CompileTeamData(team_list,my_dict,season):
    ## MAIN LOGIC
    stats_needed = [('era','pitching'),('avg','hitting'),('wins','pitching'),('obp', 'hitting'),('slg','hitting'),('ops','hitting'),('stolenBases','hitting'),('leftOnBase','hitting'),('runs','hitting'),('whip','pitching'),('runs','pitching'),('blownSaves','pitching'),('strikeoutWalkRatio','pitching')]
    for my_stat in stats_needed:
        dic = getStat(stat=my_stat[0],group=my_stat[1],season=season)
        for team_id in team_list:
            ## separate mid-season wins from prediction label
            if my_stat[0] =='wins':
                my_dict[season*1000+team_id]['cur_wins'] = dic[team_id]
            ## separate runs against from runs earned
            elif my_stat[0]=='runs' and my_stat[1]=='pitching':
                my_dict[season*1000+team_id]['runs_against'] = dic[team_id]
            ## normal case
            else: my_dict[season*1000+team_id][my_stat[0]] = dic[team_id]
    stats = statsapi.get("standings", {'season' : season, 
                                       'sportIds':1,
                                       'leagueId': "103,104"})
    for division in stats['records']:
        for team in division['teamRecords']:
            my_dict[season*1000+team['team']['id']]['wins'] = team['wins']
    ###############

    return my_dict

def ExporttoCSV(data):
    """
    Takes a nested dictionary of obvs, features as a parameter in the form:
    {obs : {feature name: feature value}}. Outputs a labeled csv file using Pandas
    """

    # Convert  dictionary to a DataFrame.
    # Using orient='index' means each key becomes a row index.
    df = pd.DataFrame.from_dict(data, orient='index')

    df.index.name = "team_id"

    # Write the DataFrame to a CSV file.
    df.to_csv("mlb_stats.csv")


## main
season_list=[]
for i in range(2013,2025):
    if i!=2020:
        season_list.append(i)

# get teamids
team_list = getTeamIds().values()
# init dic with sample labels
my_dict = {season*1000+team: {} for team in team_list for season in season_list}
for i in range(2013,2025):
    if i==2020:
        continue
    CompileTeamData(team_list,my_dict,i)
ExporttoCSV(my_dict)