#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[30]:


data=pd.read_csv('diabetes.csv')
data.head()


# In[4]:


data.describe().T


# In[31]:


data_cols=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
(data[data_cols]==0).sum()


# In[32]:


data.isnull().sum()


# In[33]:


from numpy import nan
data[data_cols]=data[data_cols].replace(0,nan)


# In[34]:


data.isnull().sum()


# In[35]:


data[data_cols].hist(figsize=(20,10))


# In[36]:


mean_val=data[data_cols].mean()
mean_val


# In[37]:


median_val=data[data_cols].median()
median_val


# In[38]:


for col in data_cols:
    data[col].fillna(data[col].median(),inplace=True)


# In[39]:


data.isnull().sum()


# In[40]:


data.info()


# In[41]:


data['Outcome']=data['Outcome'].astype('category')


# In[42]:


data.info()


# In[43]:


data.groupby(["Outcome"]).count()


# In[44]:


num_obs=len(data)
num_true=len(data.loc[data['Outcome']==1])
num_false=len(data.loc[data['Outcome']==0])
print("No of true outcomes: {0} ({1:2.2f}%".format(num_true,(num_true/num_obs)*100))
print("No of false outcomes: {1} ({1:2.2f}%".format(num_false,(num_false/num_obs)*100))


# In[45]:


data.hist(figsize=(20,10))


# In[46]:


sns.pairplot(data,hue='Outcome')
plt.show()


# In[47]:


# BOX PLOT


# In[48]:


num_f=data.select_dtypes(include='number')
col_names=num_f.columns


# In[49]:


fig=plt.figure(1,(10,5))
sns.set_palette(sns.color_palette("Accent"))
for i,cont in enumerate(col_names):
    ax=plt.subplot(3,3,1+i)
    sns.boxplot(data=data,x=data[cont],showmeans=True)
    ax.set_title(f"Distribution of {cont}")
    ax.set_xlabel("")
plt.tight_layout()


# In[50]:


#VIOLIN PLOT


# In[51]:


fig=plt.figure(i,(20,14))
sns.set_palette(sns.color_palette("Accent"))
for i,cont in enumerate(col_names):
    ax=plt.subplot(3,3,1+i)
    sns.violinplot(x='Outcome',y=cont,data=data,palette="muted",split=True)
    ax.set_title(f"Distribution of {cont}")
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
plt.tight_layout()


# In[52]:


for i,col in enumerate(data.columns[:-1]):
    sns.catplot(x='Outcome',y=col,data=data,kind="box")
plt.show()


# In[53]:


sns.countplot(x="Pregnancies",data=data,color='green')


# In[54]:


Q1=data.quantile(0.25)
Q2=data.quantile(0.75)
Q3=data.quantile(0.5)
IQR=Q2-Q1
Maximum= Q3 + 1.5 * IQR

Maximum


# In[55]:


cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI']


# In[56]:


for col in cols:
    data[col][data[col]> Maximum[col]]=data[col].median()


# In[57]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled=pd.DataFrame(scaler.fit_transform(data.iloc[: , :-1]),columns=data.columns[:-1])


# In[58]:


scaled


# In[59]:


scaled.hist(figsize=(20,10))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




