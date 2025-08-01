import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('M6/employees.csv')

# print(data.head())

data["Start Date"] = pd.to_datetime(data["Start Date"])

# print(data.head())

data["Before 2000"] = data["Start Date"].dt.year < 2000

# print(data.head())

grouped_data = data.groupby(["Gender", "Before 2000"])["Salary"].mean()


plot_data = grouped_data.unstack()  

print(plot_data)

plot_data.plot(kind="bar")

plt.title("Average Salary by Gender and Start Date")
plt.ylabel("Average Salary")
plt.xlabel("Gender")
plt.xticks(rotation=0)
plt.legend(title="Before 2000")
plt.show()
