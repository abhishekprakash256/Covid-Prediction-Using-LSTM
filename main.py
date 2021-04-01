#imports for the file
import pandas as pd









#----------------------------------------------import of the data set in the file-----------------------------------------
data_1 = pd.read_csv("archive/us_covid19_daily.csv")
data_2 = pd.read_csv("archive/us_counties_covid19_daily.csv")
data_3 = pd.read_csv("archive/us_states_covid19_daily.csv")



#-------------------------------------------------------------------------data set visualization--------------------------------

#print("data_1")
#print(data_1.head())
#print(data_1.describe())
#print("data_2")
#print(data_2.head())
#print(data_2.describe())
print("data_3")
print(data_3.head())
print(data_3.describe())

'''print("----------------------------columns in the data_1-------------------------")
for col in data_1:
    print(col)'''


'''print("----------------------------columns in the data_2--------------------------")

for col in data_2:
    print(col)'''

print("----------------------------columns in the data_3-------------------------")

for col in data_3:
    print(col)
