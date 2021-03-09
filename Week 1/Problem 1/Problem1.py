from datetime import datetime, date
import pandas as pd
import itertools


def create_date(row):
         return datetime.strptime('{:04d}/{:02d}/{:02d}'.format(row.Year, row.Month, row.Day), '%Y/%m/%d')

def load_bom(): 
    rain = pd.read_csv('Week 1\\Problem 1\\BOM\\IDCJAC0009_040913_1800_Data.csv')
    temp = pd.read_csv('Week 1\\Problem 1\\BOM\\IDCJAC0010_040913_1800_Data.csv')
    solar = pd.read_csv('Week 1\\Problem 1\\BOM\\IDCJAC0016_040913_1800_Data.csv')

    rain = rain[(rain.Year >= 2014) & (rain.Year <= 2018)]
    temp = temp[(temp.Year >= 2014) & (temp.Year <= 2018)]
    solar = solar[(solar.Year >= 2014) & (solar.Year <= 2018)]

    rain['Date'] = rain.apply(create_date, axis=1)
    temp['Date'] = temp.apply(create_date, axis=1)
    solar['Date'] = solar.apply(create_date, axis=1)

    bom = pd.merge(rain, temp, on="Date")
    bom = pd.merge(bom, solar, on='Date')

    bom = bom.drop(
        ['Year', 'Month', 'Day', 
        'Year_x', 'Month_x', 'Day_x', 
        'Year_y', 'Month_y', 'Day_y', 
        'Quality_x', 'Quality_y', 
        'Product code', 'Product code_x', 'Product code_y',
        'Bureau of Meteorology station number',
        'Bureau of Meteorology station number_x', 
        'Bureau of Meteorology station number_y',
        'Period over which rainfall was measured (days)',
        'Days of accumulation of maximum temperature'],
        axis=1)

    bom.to_csv('BOM_data.csv')

    return bom


def load_bcc():
    bcc2014 = pd.read_csv('Week 1\\Problem 1\\BCCCyclewayCounts\\bike-ped-auto-counts-2014.csv')
    bcc2015 = pd.read_csv('Week 1\\Problem 1\\BCCCyclewayCounts\\bike-ped-auto-counts-2015.csv')
    bcc2016 = pd.read_csv('Week 1\\Problem 1\\BCCCyclewayCounts\\bike-ped-auto-counts-2016.csv')
    bcc2017 = pd.read_csv('Week 1\\Problem 1\\BCCCyclewayCounts\\bike-ped-auto-counts-2017.csv')
    bcc2018 = pd.read_csv('Week 1\\Problem 1\\BCCCyclewayCounts\\bike-ped-auto-counts-2018.csv')

    bcc2014.Date = pd.to_datetime(bcc2014.Date, format='%d/%m/%Y')
    bcc2015.Date = pd.to_datetime(bcc2015.Date, format='%d/%m/%Y')
    bcc2016.Date = pd.to_datetime(bcc2016.Date, format='%d/%m/%Y')
    bcc2017.Date = pd.to_datetime(bcc2017.Date, format='%d/%m/%Y')
    bcc2018.Date = pd.to_datetime(bcc2018.Date, format='%d/%m/%Y')

    # Find all distinct column headers
    all_columns = set(itertools.chain(*(
        bcc2014.columns.values, bcc2015.columns.values,
        bcc2016.columns.values, bcc2017.columns.values,
        bcc2018.columns.values)))

    # Find all common column headers
    common_columns = all_columns.intersection(
        bcc2014.columns.values, bcc2015.columns.values,
        bcc2016.columns.values, bcc2017.columns.values,
        bcc2018.columns.values)
    
    # Get common columns
    bcc2014 = bcc2014[common_columns]
    bcc2015 = bcc2015[common_columns]
    bcc2016 = bcc2016[common_columns]
    bcc2017 = bcc2017[common_columns]
    bcc2018 = bcc2018[common_columns]

    # Combine vertically (Year asc)
    bcc = pd.concat([bcc2014, bcc2015, bcc2016,
                    bcc2017, bcc2018])
                
    # Drop unnecessary info
    bcc = bcc.drop("Unnamed: 1", axis=1)

    bcc.to_csv('BCC_data.csv')

    return bcc
    


bom = load_bom()
bcc = load_bcc()

combined = pd.merge(bom, bcc, on='Date')
combined.to_csv('combined.csv')




# Cemetary
# rainHead = rain.head(1).Year.iat[0]
# rainTail = rain.tail(1).Year.iat[-1]

# tempHead = temp.head(1).Year.iat[0]
# tempTail = temp.tail(1).Year.iat[-1]

# solarHead = solar.head(1).Year.iat[0]
# solarTail = solar.tail(1).Year.iat[-1]

# maxYear = min(rainTail, tempTail, solarTail)
# minYear = max(rainHead, tempHead, solarHead)