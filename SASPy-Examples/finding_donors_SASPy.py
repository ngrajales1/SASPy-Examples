# This is an example of how SASpy can be used within python to predict potential donors from US Census data.


# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.

#Import necessary libraries for this project
import saspy
import pandas as pd
from time import time
from IPython.display import display
from IPython.display import HTML

# starting the SAS session
sas = saspy.SASsession(cfgname='autogen_winlocal')

#loading the census data.

#The foloowing method below display how to read a csv file using pandas and then reading in
#the data frome into a SAS object
#cen_data0_pd = pd.read_csv("C:\\Users\\negraj\\Documents\\Learning\\SAS and SASPy\\SASPy-Examples\\census.csv")
#cen_data0 = sas.df2sd(cen_data0_pd) # the short form of: hr = sas.dataframe2sasdata(hr_pd)

#You can also read in a dataset directly into a SAS object using the code below.
cen_data0 = sas.read_csv("C:\\Users\\negraj\\Documents\\Learning\\SAS and SASPy\\SASPy-Examples\\census.csv")

#Display the first record to verify data was read in
#display(cen_data0_pd.head(n=1))
print("First value is:")
display(cen_data0.head(obs=1))

#Data Exploration using SASpy
#display the amount of observations within your data
records = cen_data0.obs()

#Number of records where individuals income is more than $50,000
# I am first going to turn our SAS data object into a pandas data frame to utilize
#value counts. Using value counts allows me to count distnict values very easily
cen_data0_pd = cen_data0.to_df()

#Using value counts to count ditninct values in the data set. A quick print of the value displays the
# ditinct value and its count. The first value returned by value counts is values x <= 50K
# the second value is values x > 50K
greater_50k = cen_data0_pd['income'].value_counts()[1]
_50k_or_less = cen_data0_pd['income'].value_counts()[0]

# Determining the percent of individuals whose income is greater than 50k
greater_percent = (greater_50k/(greater_50k+_50k_or_less))*100

print("Total number of records: {}".format(records))
print("Individuals making more than $50,000: {}".format(greater_50k))
print("Individuals making at most $50,000: {}".format(_50k_or_less))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))

#----------------------Part 2------------------------------
# Featureset Exploration
#Transforming Skewed continous features
#You need to use the content function to view the column number to then use it in the histogram function later

#drop income using pandads DataFrame
income_data = cen_data0_pd['income']
feature_data = cen_data0_pd.drop('income', axis=1)

#convert back to SAS data object
cen_data1_noIncome = sas.dataframe2sasdata(feature_data)

#diplay histogram for capital gains and capital loss
for col in ['capital-gain','capital-loss']:
    cen_data1_noIncome.hist(col, title='Histogram showing'+col.upper())

#As the histogram displays above the results on capital gains are highly skewed. This can cause issues later on
# when trying to make decisons based on this data. To resolve this issue we can Mormalize the features using different
#methods. We will first try log based methods and then a normailzed one.

#log based method



print("End of program")
