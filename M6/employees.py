import pandas as pd

data = pd.read_csv('M6/employees.csv')

# print(data.head())

data["Start Date"] = pd.to_datetime(data["Start Date"])

# print(data.head())

data["Before 2000"] = data["Start Date"].dt.year < 2000

print(data.head())

grouped_data = data.groupby(["Gender", "Before 2000"])["Salary"].mean()

print(grouped_data)