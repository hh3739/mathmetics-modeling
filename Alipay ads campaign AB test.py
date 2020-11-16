'''
A real-world dataset collected from two marketing campaigns in Alipay.




This dataset contains two commercial targeting campaign logs in Alipay. Due to privacy issue, data is sampled and desensitized. 
Although the statistical results on this data set deviate from the actual scale of Alipay.com, 
it will not affect the applicability of the solution.

emb_tb_2.csv: User feature dataset.
effect_tb.csv: Click/Non-click dataset.
seed_cand_tb.csv: Seed users and candidate users dataset.


dmp_id	The unique ID of a targeting campaign.
user_id	The unique ID of an Alipay user.
label	Denotes whether a user clicked the campaign ads in that day dt.
dt	Values from {1,2}. Indicates whether it's a first day log (“1”) or a second day log (“2”) for the target campaign.

'''
import pandas as pd 
from pandas import Series,DataFrame
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set_style('darkgrid')


df = pd.read_csv('./audience_expansion/effect_tb.csv')
df.columns = ["dt","user_id","label","dmp_id"]
df.head(10)


df.info(null_counts=True)



# group by campaign id, and statistic out the click rate of each campaign 

df.drop('dt',axis=1,inplace=True)

# checking duplicated value 
df[df.duplicated()].sort_values('user_id',ascending=True)
df.drop_duplicates(inplace=True)

# checking outlier using pivot_table 

df_pf = pd.pivot_table(data=df, index='dmp_id',columns='label',values='user_id',aggfunc='count',margins=True)
display(df_pf)
# there is no outlier value 

# find out the mean value of each campaign 
df.groupby('dmp_id').mean()
# obvious campign3  has 2.6% of click rate , we can tell campaign3 has higher click rate than campaign1, however 
# we need to use hypothesis test to test the population mean of two campaigns 

# In this case, we are going to use ratio test because the distribution of sample is Binomial distribution 
# examing the preconditions for ratio test 
shape = df['dmp_id'].value_counts()
n1 = shape[1]
p1 = df['label'][df['dmp_id']==1].mean()

n3 = shape[3]
p3 = df['label'][df['dmp_id']==3].mean()

if n1*p1>5 and n3*p3>5 and n1*(1-p1)>5 and n3*(1-p3)>5:
    print('ratio test is fulfillable')
else:
    print('fail') 

# this is two populations raito test, so Z statistics =(p1 - p2)/np.sqrt(p*(1-p)/(n1+n2))

from scipy import stats 

# Null Hypothesis: ratio of campagin3 - ratio of campagin1 <= 0 , alternative Hypothesis : roc3 - roc1 > 0 , this is right 
# side hypothesis test 

p = (n1*p1+n3*p3)/(n1+n3)

Z= (p3-p1)/np.sqrt(p*(1-p)/(n1+n3))
Probablity = stats.norm.sf(Z)
print(Probablity)

# As probablity is very close to 0%, reject null hypothesis so we strongly believe that campaign3's click rate 
# are higher than campaign 1 . 

