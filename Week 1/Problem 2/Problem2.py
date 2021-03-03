import pandas as pd
import numpy as np

data = pd.read_csv('Week 1\\Problem 2\\combined_example.csv')

rowlen = len(data.columns.values)

for column in data.columns.values:
    test = data[column]
    print(column)

    # if np.sum(column)

# def removeNullColumns:
    
    # for(column in data.colum)

test = data.head().columns.values

print("F")