#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np                                        # importing necessary libraries
import math
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import datetime
from IPython.display import set_matplotlib_formats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_excel("InputData.xlsx")   # Reading the Data


# In[3]:


IE = 100000                          # Initial Equity ( Assumed to be 100k)
E_GE = IE/4                          # distributing equal dollars to all the asset classes
E_HF = IE/4
E_RP = IE/4
E_TF = IE/4


# In[4]:


df['GE'] = np.exp(df['Global Equities'])*E_GE # using log(today/previous) definintion of return and then evaluating value of all asset classes after incorporating return 
df['HF'] = np.exp(df['Hedge Funds'])*E_HF     
df['RP'] = np.exp(df['Risk Premia'])*E_RP
df['TF'] = np.exp(df['Trend Following'])*E_TF


# In[5]:


for i in range(1,len(df)):                     # applying a loop over the length of dataframe and updating the values (eg 100 as initial value ( 3percent return make it 103 and the 5percent return in next month make it 108.15)
    df['GE'][i] = np.exp(df['Global Equities'][i])*df['GE'][i-1]     
    df['HF'][i] = np.exp(df['Hedge Funds'][i])*df['HF'][i-1]
    df['RP'][i] = np.exp(df['Risk Premia'][i])*df['RP'][i-1]
    df['TF'][i] = np.exp(df['Trend Following'][i])*df['TF'][i-1]


# In[6]:


df['PV'] = df['GE'] + df['HF'] + df['RP'] + df['TF']  # Portfolio Value
df['Date'] = df['Date'].astype('datetime64[ns]')    # conversion of Date to datetime format
df['month'] = df['Date'].dt.month                   


# In[7]:


q =  np.array([3,6,9,12])  # array for storing the quarter months


# In[8]:


def quartely_rebalancing(df):
    p=0
    for index, row in df.iterrows():                 # Quartely Rebalancing
        if (p==len(df)-1):                           
            break
        if(df['month'][p] in q):
            IE=df['PV'][p]
            E_GE = IE/4
            E_HF = IE/4
            E_RP = IE/4
            E_TF = IE/4
            df['GE'][p+1] = np.exp(df['Global Equities'][p+1])*E_GE
            df['HF'][p+1] = np.exp(df['Hedge Funds'][p+1])*E_HF
            df['RP'][p+1] = np.exp(df['Risk Premia'][p+1])*E_RP
            df['TF'][p+1] = np.exp(df['Trend Following'][p+1])*E_TF
            for i in range(2,4):
                df['GE'][p+i] = np.exp(df['Global Equities'][p+i])*df['GE'][p+(i-1)]
                df['HF'][p+i] = np.exp(df['Hedge Funds'][p+i])*df['HF'][p+(i-1)]
                df['RP'][p+i] = np.exp(df['Risk Premia'][p+i])*df['RP'][p+(i-1)]
                df['TF'][p+i] = np.exp(df['Trend Following'][p+i])*df['TF'][p+(i-1)]
            for i in range(1,4):
                df['PV'][p+i] = df['GE'][p+i] + df['HF'][p+i] + df['RP'][p+i] + df['TF'][p+i]
                
        p=p+1
    
    return df
            
            


# In[9]:


asset_balance = quartely_rebalancing(df) # dataframe which stores quartely rebalanced portfolio values


# In[10]:


asset_balance['Portfolio Return'] = np.log(asset_balance['PV']/asset_balance['PV'].shift(1)) # computing portfolio return


# In[11]:


def portfolio_statistics(df):
    annualised_geometric_port_return = np.prod(df['Portfolio Return'] + 1) **(12/df['Portfolio Return'].shape[0]) - 1
    annual_std = np.std(df['Portfolio Return']) * np.sqrt(12)  
    port_sharpe_ratio = annualised_geometric_port_return / annual_std
    return annualised_geometric_port_return,annual_std,port_sharpe_ratio
    


# In[12]:


annualised_geometric_port_return,annual_std,port_sharpe_ratio = portfolio_statistics(asset_balance)


# In[13]:


print("value of annualized return is "+"{:.2%}".format(annualised_geometric_port_return));
print("value of annualized standard deviation is "+"{:.2%}".format(annual_std));
print("value of sharpe ratio is %f  "%port_sharpe_ratio)


# In[14]:


def cummalative_ret(asset_balance):
    cumulative_ret = (asset_balance['Portfolio Return'] + 1).cumprod()
    date = asset_balance['Date']
    plt.plot(date,cumulative_ret)
    plt.xlabel('Date')
    plt.ylabel("Cumulative Returns")
    plt.title("Portfolio Cumulative Returns")
    plt.savefig('Portfolio Cumulative Returns.png')
    plt.show();
    


# In[15]:


cummalative_ret(asset_balance)


# In[16]:


def portfolio_performance(asset_balance):
    asset_balance_2020 = asset_balance.loc[asset_balance['Date'].dt.year==2020] # Dataframe for 2020 YTD
    asset_balance_last_one_year = asset_balance.set_index('Date').last('12M')  # Dataframe for last one year
    asset_balance_last_three_year = asset_balance.set_index('Date').last('36M') # Dataframe for last three years
    asset_balance_2020=asset_balance_2020.drop(['Global Equities', 'Hedge Funds','Risk Premia', 'Trend Following','month','Risk Free Returns','Portfolio Return'], axis=1)
    asset_balance_last_one_year=asset_balance_last_one_year.drop(['Global Equities', 'Hedge Funds','Risk Premia', 'Trend Following','month','Risk Free Returns','Portfolio Return'], axis=1)
    asset_balance_last_three_year=asset_balance_last_three_year.drop(['Global Equities', 'Hedge Funds','Risk Premia', 'Trend Following','month','Risk Free Returns','Portfolio Return'], axis=1)
    return asset_balance_2020,asset_balance_last_one_year,asset_balance_last_three_year


# In[17]:


asset_balance_2020,asset_balance_last_one_year,asset_balance_last_three_year=portfolio_performance(asset_balance)


# In[18]:


# contribution of different asset classes in 2020YTD, last one year and last three year
def contribution_asset_classes(asset_balance):
    sum_asset = asset_balance.sum(axis = 0, skipna = True)
    net_contribution = [sum_asset[0],sum_asset[1],sum_asset[2],sum_asset[3]]
    arry_storing = [sum_asset[0]/sum_asset[4],sum_asset[1]/sum_asset[4],sum_asset[2]/sum_asset[4],sum_asset[3]/sum_asset[4]]
    return arry_storing, net_contribution


# In[19]:


proportion_asset_classes_2020,net_contribution_asset_classes_2020= contribution_asset_classes(asset_balance_2020)
print("Contribution of Global Equities in 2020 is "+"{:.2%}".format(proportion_asset_classes_2020[0]))
print("Contribution of Hedge Funds in 2020 is "+"{:.2%}".format(proportion_asset_classes_2020[1]))
print("Contribution of Risk Premia in 2020 is "+"{:.2%}".format(proportion_asset_classes_2020[2]))
print("Contribution of Trend Following in 2020 is "+"{:.2%}".format(proportion_asset_classes_2020[3]))


# In[20]:


proportion_asset_classes_last_one_year,net_contribution_asset_classes_last_one_year= contribution_asset_classes(asset_balance_last_one_year)
print("Contribution of Global Equities in last one year is "+"{:.2%}".format(proportion_asset_classes_last_one_year[0]))
print("Contribution of Hedge Funds in last one year is "+"{:.2%}".format(proportion_asset_classes_last_one_year[1]))
print("Contribution of Risk Premia in last one year is "+"{:.2%}".format(proportion_asset_classes_last_one_year[2]))
print("Contribution of Trend Following in last one year is "+"{:.2%}".format(proportion_asset_classes_last_one_year[3]))


# In[21]:


proportion_asset_classes_last_three_year,net_contribution_asset_classes_last_three_year= contribution_asset_classes(asset_balance_last_three_year)
print("Contribution of Global Equities in last three year is "+"{:.2%}".format(proportion_asset_classes_last_three_year[0]))
print("Contribution of Hedge Funds in last three year is "+"{:.2%}".format(proportion_asset_classes_last_three_year[1]))
print("Contribution of Risk Premia in last three year is "+"{:.2%}".format(proportion_asset_classes_last_three_year[2]))
print("Contribution of Trend Following in last three year is "+"{:.2%}".format(proportion_asset_classes_last_three_year[3]))


# In[22]:


asset_balance = asset_balance.drop(['month'],axis=1)
asset_balance.to_csv('portfolio_log_.csv')


# In[23]:


#### Improving the construction of portfolio


# In[24]:


optim_port = pd.read_excel("InputData.xlsx") # Inputing the data again


# In[25]:


optim_port= optim_port.drop(['Risk Free Returns','Date'],axis=1) 


# In[26]:


mean_daily_returns = optim_port.mean() # finding mean daily returns and covariance matrix
cov_matrix = optim_port.cov()         


# In[27]:


M = mean_daily_returns.to_numpy()  # Applying sharpe theory of portfolio optimization and evaluating necessary terms
MT = M.reshape((M.shape[0], 1))
MT = MT * 12  # conversion to annualized returns
#MT.shape
M = MT.T
#M.shape
o = np.ones((1,4), dtype = int)
#o.shape
cov = cov_matrix 
cov = np.asarray(cov)
#cov.shape
cov_in = np.linalg.inv(cov)


# In[28]:


u_rf = df['Risk Free Returns'].mean()*12        # annualized risk free returns


# In[29]:


W1_num = np.dot((M-np.dot(u_rf,o)),cov_in)
W1_den = np.dot((M-np.dot(u_rf,o)),np.dot(cov_in,o.T))  # W1* (optimized matrix of weights (w/w_risky) )
W1 = W1_num/W1_den


# In[30]:


u_der = np.dot(M,W1.T)
u_der.shape
sigma_square  = np.dot(W1,np.dot(cov,W1.T)) * 12
sigma_der = np.sqrt(sigma_square)
sigma_der.shape
s = (u_der - u_rf)/(sigma_der)
u = u_rf + s*np.sqrt(sigma_square)


# In[31]:


point1 = [0, sigma_der[0][0]]
point2 = [u_rf, u_der[0][0]]
plt.plot(point1, point2, label = "line 1")
plt.xlabel('Systematic Risk')
plt.ylabel(" Expected Return")
plt.title("Capital Market Line")
plt.savefig('Capital Market Line.png')


# In[32]:


print("The relation between return and risk is : u = %r*sigma + %r  "%(s[0][0],u_rf))


# In[33]:


# using the above obtained relation and assuming proportion of risk-free assets in our portfolio we can get optimized weights  

