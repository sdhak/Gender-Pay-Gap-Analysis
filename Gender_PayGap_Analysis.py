#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install datascience


# In[27]:


from datascience import *
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg', warn=False)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[28]:


survey_schema = pd.read_csv("/Users/shristidhakal/Documents/Grad School/INFO5502/Gender_pay_gap/SurveySchema.csv")
survey_schema.head(2)


# In[29]:


freeForm_responses = pd.read_csv("/Users/shristidhakal/Documents/Grad School/INFO5502/Gender_pay_gap/freeFormResponses.csv")
freeForm_responses.head(2)


# In[30]:


MCQ = pd.read_csv("/Users/shristidhakal/Documents/Grad School/INFO5502/Gender_pay_gap/multipleChoiceResponses.csv")
MCQ.head(3)


# In[31]:


#Data Cleaning
income = MCQ[['Q1', 'Q2', 'Q9']]

columns = ['Gender', 'Age', 'Income'] 
income = income[1:] 
income.columns = columns

gender_income = income[['Gender', 'Income']]
gender_income = gender_income[gender_income['Income'].notnull()]
gender_income = gender_income[gender_income['Income'] != 
                              'I do not wish to disclose my approximate yearly compensation']
gender_income = gender_income[gender_income['Gender'] != 'Prefer not to say']
gender_income = gender_income[gender_income['Gender'] != 'Prefer to self-describe']

calc_inc = gender_income["Income"].str.split("-", n = 1, expand = True) 
calc_inc[0] = calc_inc[0].map(lambda x: x.rstrip('+')) 
calc_inc[0] = calc_inc[0].str.replace(',', '')
calc_inc[1] = calc_inc[1].str.replace(',', '')

calc_inc[[0, 1]] = calc_inc[[0, 1]].apply(pd.to_numeric, errors='coerce')
calc_inc[0] = calc_inc[0] * 1000

calc_inc['Income'] = calc_inc[[0,1]].mean(axis=1)

calc_inc.loc[calc_inc['Income'] > 499999999, 'Income'] = 500000


# In[32]:


#Calculating median income of male and female employees
gender_income['Income'] = calc_inc['Income']
median_income = gender_income.groupby('Gender')['Income'].median().to_frame().reset_index()
median_income.head()


# In[33]:


#Exporting gender & income data to a new table so we can
#use this simple table for rest of the assignment.
gender_income.to_csv(
    "/Users/shristidhakal/Documents/Grad School/INFO5502/Gender_pay_gap/gender_income.csv")

income = Table.read_table(
    "/Users/shristidhakal/Documents/Grad School/INFO5502/Gender_pay_gap/gender_income.csv")
income = income.select('Gender', 'Income')
income


# In[34]:


#histogram of median incomes by gender ($0-$500,000; 50,000 units per bin)
bins = np.arange(0,500000,50000)
income.hist('Income', group = 'Gender', bins = bins)


# In[35]:


#A closer look at the histogram (Income $0-$300,000; 20,000 units per bin)
bins = np.arange(0,300000,20000)
income.hist('Income', group = 'Gender', bins = bins)


# In[36]:


#A closer look of the income higher than $150,000.. 
#..(Income $150,000-$500,000; 30,000 units per bin)
newbins = np.arange(150000,500000,30000)
income.hist('Income', group = 'Gender', bins = newbins)


# In[37]:


#Random sampling of 500 employees.
#To generate a fair random sampling of males and females, we need to identify..
#..the ratio of male and ratio of female out of total population in the dataset, and..
#..generate sample based on their proportion

female = income.where('Gender', 'Female')
male = income.where('Gender', 'Male')
print(np.round((female.num_rows/income.num_rows)*100, decimals=1), 
      "% female,", np.round((male.num_rows/income.num_rows)*100, decimals=1), "% male")


# In[38]:


female_to_male = [0.156, 0.844]

def sample_f():
  return (100 * sample_proportions(500, female_to_male)).item(0)

counts = make_array()

repetitions = 10000
for i in np.arange(repetitions):
    counts = np.append(counts, sample_f())

Table().with_column('Sample Female Count', counts).hist(bins = np.arange(10, 25, 1))
plt.scatter(15.6, 0, color='red', s=50);

#Below is a histogram showing evidence that this sampling generates fair samples. 
#No. of females shown in the histogram lie roughly between 11-21..
#..with a red dot showing the actual proportion of females in the data set. 


# In[39]:


#Test statistic: Difference between male and female mean income
#Null Hypothesis: There is no difference between the mean income of male and female.
#Alternative Hypothesis: There is a statistical difference between the..
#..mean income of male and mean income of female.


# In[41]:


bins = np.arange(10000,200000,20000)

def total_sample(n):
    sample = income.sample(n)
    sample.hist(group = 'Gender', bins = bins)
    sample_median = sample.column('Income')
    print("The median income of the sample is",np.median(sample_median),"\n")

total_sample(500)

plt.scatter(25000,0, s=100, c='y')
plt.scatter(35000,0, s=100, c='r')


# In[42]:


def sample_statistic():
    sample = income.sample(5000)
    samp_male = sample.where('Gender', 'Male')
    samp_male = samp_male.column('Income')
    samp_female = sample.where('Gender', 'Female')
    samp_female = samp_female.column('Income')

    sample_test_statistic = np.mean(samp_male) - np.mean(samp_female)
    
    return sample_test_statistic


# In[43]:


pop_male = income.where('Gender', 'Male')
pop_male = pop_male.column('Income')
pop_female = income.where('Gender', 'Female')
pop_female = pop_female.column('Income')

pop_test_statistic = np.mean(pop_male) - np.mean(pop_female)
print("The difference of the mean incomes between female and male employees is $",np.round(pop_test_statistic,2), ".")


# In[44]:


#Permutation test of samples: 10,000 times.
#Histogram of the sample test statistic repeated 10,000 times
count = make_array()

simulations = 10000
for i in np.arange(simulations):
    count = np.append(count, sample_statistic())

count_table = Table().with_column('Differences between Mean Incomes of the Sample', count)
count_table.hist(bins = np.arange(-2000, 15000, 1000))
plt.scatter(pop_test_statistic, 0, color='red', s=100);


# In[52]:


#Bootstrap samples.

sample = count_table.sample(5000)

def bootstrap_mean(sample, label, replications):
    one_column = sample.select(label)
    mean = make_array()
    for i in np.arange(replications):
        bootstrap_sample = one_column.sample()
        resampled_mean = np.mean(bootstrap_sample.column(0))
        mean = np.append(mean, resampled_mean)
        
    return mean

bootstrap = bootstrap_mean(sample, 'Differences between Mean Incomes of the Sample', 5000)


# In[50]:


resampled_medians = Table().with_column('Bootstrap Sample Mean', bootstrap)
resampled_medians.hist()

plt.scatter(pop_test_statistic, 0, color='red', s=100);


# In[53]:


#We look at the middle 95% of the bootstrap sample means.
lbound = np.round(percentile(2.5, bootstrap), 2)
rbound = np.round(percentile(97.5, bootstrap), 2)

make_array(lbound, rbound)


# In[54]:


#histogram of the bootstrap sample statistic

resampled_medians.hist()
plt.plot(make_array(lbound, rbound), make_array(0, 0), color='yellow', lw=3);
plt.scatter(pop_test_statistic, 0, color='red', s=100);


# In[ ]:




