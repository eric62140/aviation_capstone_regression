#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[3]:


#read data and show the first five lines of data
db = pd.read_csv('BWIFullOutageSelect_20181005.csv')
db = db[db['Log_Id,N,10,0'] != 0]
#rename each column 
db = db.rename(columns = {'Log_Id,N,10,0':'Log_id','Fac_Type,C,254':'Fac_type','Fac_Ident,C,254':'Fac_ident','Fac_Code,C,254':'Fac_code','Runway,C,254':'Runway','Latitude_D,N,19,11':'Latitude','Longitude_,N,19,11':'Longitude','Assoc_Airp,C,254':'Assoc_airp','Code_Categ,N,10,0':'Code_categ','Interrupt_,C,254':'Interrupt','Supplement,C,254':'Supplement','Sup_Codede,C,254':'Sup_code','Maint_Acti,C,254':'Maint_acti','Mac_Descri,C,254':'Mac_descri','Start_Date,D':'Start_date','End_Dateti,D':'End_date','Log_Summar,C,254':'Log_summary'})
#copy data and drop useless columns
db = db.drop(['Fac_code','Runway','Assoc_airp','Log_summary','Latitude','Longitude'], axis=1)
db['Fac'] = db.Fac_type + '-' + db.Fac_ident
db = db.drop(['Fac_type','Fac_ident'],axis = 1)
db['Start_date'] = pd.to_datetime(db['Start_date'],infer_datetime_format=True)
db['End_date'] = pd.to_datetime(db['End_date'],infer_datetime_format=True)
db['Outage_dur'] = db.End_date - db.Start_date
dbb = db.drop(db.index[0:21])


# In[22]:


De = pd.read_csv('delayed_per_day.csv')
eqt = pd.read_csv("eqt_delay.csv")
De['Date'] = pd.to_datetime(De['Date'], errors='coerce')#.dt.date
De['average_delay'] = De[['depart', 'arrive']].mean(axis=1)
De['isoutage'] = 0
dbb['eqt_delay'] = 0
dbb['eqt_min'] = 0
eqt['delay_eqt'] = 0
eqt['Date'] = pd.to_datetime(eqt['Date'], format='%Y/%m/%d')


# In[24]:


for date in dbb['Start_date']:
    for date2 in De['Date']:
        if (date2==date):
            currentDate = datetime.strptime(date2.strftime('%Y/%m/%d'),'%Y/%m/%d').date()
            mask = (De.Date == currentDate)
            column_name = 'isoutage'
            De.loc[mask, column_name] = 1


# In[25]:


for date in dbb['Start_date']: 
    for date2 in eqt['Date']:  
        if (date2==date): 
            s = eqt[eqt['Date']==date2]['Min']
            for i in s:
                if i>0:
                    
                    mask = (eqt.Date == date2)
                    mask2 = (dbb.Start_date == date2)
                    column_name = 'delay_eqt'
                    column_name2 = 'eqt_delay'
                    column_name3 = 'eqt_min'
                    eqt.loc[mask, column_name] = 1
                    dbb.loc[mask2, column_name2] = 1
                    dbb.loc[mask2, column_name3] = i


# In[30]:


newdata = dbb[['Start_date', 'Fac', 'eqt_delay', 'eqt_min']].copy()


# In[39]:


# remove the same startdate and facility type of the outage in the same day
def takeawaysame(newdata):
    kk = newdata[:]
    pp = kk # new dataframe to use 
    pp = pp.reset_index(drop=True)
    datee = []
    facc = []
    for i in range(len(pp)):
        s = pp[i:i+1]['Start_date']
        f = pp[i:i+1]['Fac']
        for k in s:
            k = k.strftime('%Y/%m/%d')
            datee.append(k)
        for ff in f:
            facc.append(ff)
    tt = pp.copy()
    
    dateee = datee
    faccc = facc
    for i in range(1,len(pp.index)):
        if (dateee[i-1] == dateee[i]) and (faccc[i-1] == faccc[i]):
            tt.iloc[i:i+1]['Fac'] = 'NaN'
    test = tt
    test = test[test.Fac != 'NaN']
    test = test.reset_index(drop=True)
    
       
    return test


# In[41]:


cleandata = takeawaysame(newdata)


# In[42]:


# #one hot encoding
# facility = []
# for fac in cleandata['Fac']:
#     facility.append(fac)
# values = array(facility)
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded)
# x = onehot_encoded


# In[43]:


temp = cleandata.groupby(['Start_date'], as_index=False).agg({'Fac': lambda x: ', '.join(x), 'eqt_delay' : 'first', 'eqt_min':'first'})
newset = temp.set_index('Start_date').Fac.str.split(', ', expand=True).stack()
newset = pd.get_dummies(newset).groupby(level=0).sum()
y = temp[['eqt_min','Start_date']]
y = y.set_index('Start_date')


# In[44]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
regr = linear_model.LinearRegression()
regr.fit(newset, y)


# In[47]:


print(regr.predict(newset.iloc[0:15]))


# In[48]:


import statsmodels.api as sm
x2 = sm.add_constant(newset)
est = sm.OLS(y, x2)
est2 = est.fit()
print(est2.summary())


# In[ ]:




