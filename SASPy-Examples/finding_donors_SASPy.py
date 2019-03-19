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
cen_data0_pd = cen_data0.to_df()
greater_50k = cen_data0_pd['income'].value_counts()[1]

display(greater_50k)
print("greater_50k")

print("End of program")
