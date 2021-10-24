import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from scipy import stats


def analyze(data):
    ###Start: Draw Heatmap for all data
    plt.figure(figsize=(10, 10))
    seaborn.heatmap(data.corr(), annot=True, cmap="coolwarm").set_title('Correlation')
    plt.show()
    ###End: Draw Heatmap for all data

    ###Start: Calculate PointBiSerial correlation
    c1 = data.SOS_Incidence
    c2 = data.Household_receives_UCT
    c3 = data.Female_head_of_household
    print(stats.pointbiserialr(c1, c3))
    print(stats.pointbiserialr(c1, c2))
    ###End: Calculate PointBiSerial correlation


def read_file():
    ###Start: Read Data and reformat column names
    dat = pd.read_excel("Dataset.xls")
    dat.columns = [col.replace(' ', '_') for col in dat.columns]
    ###End: Read Data and reformat column names
    return dat


def factorize(fact_data):
    ###Start: Convert textual categorical values to numerical
    fact_data['Year'] = pd.factorize(fact_data.Year)[0]
    fact_data['Region'] = pd.factorize(fact_data.Region)[0]
    fact_data['Female_head_of_household'] = pd.factorize(fact_data.Female_head_of_household)[0]
    fact_data['Highest_level_of_schooling_attained_head_of_household'] = pd.factorize(fact_data.Highest_level_of_schooling_attained_head_of_household)[0]
    fact_data['Per_capita_expenditure_quintile'] = pd.factorize(fact_data.Per_capita_expenditure_quintile)[0]
    fact_data['Rural_area'] = pd.factorize(fact_data.Rural_area)[0]
    ###End: Convert textual categorical values to numerical
    return fact_data


def print_stats(statData):
    ###Start: Print Count of Values for categorical columns
    print(statData.Year.value_counts())
    print(statData.Region.value_counts())
    print(statData.Female_head_of_household.value_counts())
    print(statData.Highest_level_of_schooling_attained_head_of_household.value_counts())
    print(statData.Per_capita_expenditure_quintile.value_counts())
    print(statData.Rural_area.value_counts())
    ###End: Print Count of Values for categorical columns

a = read_file()
print_stats(a)
a = factorize(a)
##Analyze all Data
analyze(a)

# Only 2005 data for households that received UCT in 2006 but did not in 2005
b = read_file()
b = b.drop_duplicates(subset=['Household_ID', 'Household_receives_UCT'], keep=False).reset_index(drop=True)
b.drop(b[b.Year == 2006].index, inplace=True)
b = factorize(b)
analyze(b)

# Only 2005 data for households that did NOT receive UCT
c = read_file()
duplicates = c.duplicated(subset=['Household_ID', 'Household_receives_UCT'], keep=False).reset_index(drop=True)
c = c[duplicates]
c.drop(c[c.Year == 2006].index, inplace=True)
c = factorize(c)
analyze(c)

# Only 2006 data for households that received UCT in 2006 but did not in 2005
d = read_file()
d = d.drop_duplicates(subset=['Household_ID', 'Household_receives_UCT'], keep=False).reset_index(drop=True)
d.drop(d[d.Year == 2005].index, inplace=True)
d = factorize(d)
analyze(d)

# Only 2006 data for households that did NOT receive UCT
e = read_file()
duplicates = e.duplicated(subset=['Household_ID', 'Household_receives_UCT'], keep=False).reset_index(drop=True)
e = e[duplicates]
e.drop(e[e.Year == 2005].index, inplace=True)
e = factorize(e)
analyze(e)
