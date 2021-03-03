import pandas as pd

def create_date(row):
        return pd.to_datetime(row.Year*10000+row.Month*100+row.Day,format='%Y%m%d')

def load_bom(): 
    rain = pd.read_csv('BOM\\IDCJAC0009_040913_1800_Data.csv')
    temp = pd.read_csv('BOM\\IDCJAC0010_040913_1800_Data.csv')
    solar = pd.read_csv('BOM\\IDCJAC0016_040913_1800_Data.csv')

    rain = rain[(rain.Year >= 2014) & (rain.Year <= 2018)]
    temp = temp[(temp.Year >= 2014) & (temp.Year <= 2018)]
    solar = solar[(solar.Year >= 2014) & (solar.Year <= 2018)]

    rain['Date'] = rain.apply(create_date, axis=1)
    temp['Date'] = temp.apply(create_date, axis=1)
    solar['Date'] = solar.apply(create_date, axis=1)

    combined = pd.merge(rain, temp, on="Date")
    combined = pd.merge(combined, solar, on='Date')

    combined = combined.drop(['Year', 'Month', 'Day', 'Year_x', 'Month_x', 'Day_x', 'Year_y', 'Month_y', 'Day_y'], axis=1)

    combined.to_csv('BOM_data.csv')

    return combined


def load_bcc():
    bcc2014 = pd.read_csv('Week 1\\BCCCyclewayCounts\\bike-ped-auto-counts-2014.csv')

    test = bcc2014.head()

    print("test")


load_bom()
load_bcc()

    






# Cemetary
# rainHead = rain.head(1).Year.iat[0]
# rainTail = rain.tail(1).Year.iat[-1]

# tempHead = temp.head(1).Year.iat[0]
# tempTail = temp.tail(1).Year.iat[-1]

# solarHead = solar.head(1).Year.iat[0]
# solarTail = solar.tail(1).Year.iat[-1]

# maxYear = min(rainTail, tempTail, solarTail)
# minYear = max(rainHead, tempHead, solarHead)