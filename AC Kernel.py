
# coding: utf-8

# # Prediction of Costa Rican Household Poverty Prediction

# ### Data Exploration 

# In[1]:


#import functions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

#read data
train = pd.read_csv("train.csv")


# The training dataset has 9557 rows and 143 columns: 

# In[25]:


train.shape


# The following columns contain missing values: v2a1 (Monthly rent payment), v18q1 (number of tablets, household owns), rez_esc (Years behind in school), meaneduc (average years of education for adults), SQBmeaned (square of the mean years of education of adults (>=18) in the household)

# In[26]:


#count number of Nan in each column
train.isnull().sum()[train.isnull().sum()>0]


# There are 2988 unique households in the training set. 

# In[27]:


#number of unique households
len(set(train["idhogar"]))


# The training dataset has class imbalances. 7.9% (755 obs) of observations are class "Extreme", 16.7% (1597 obs) are class "Moderate", 12.7% (1209 obs) are class "Vulnerable", 62.7% (5996 obs) are class "Non-Vulnerable." 

# In[28]:


ncount = len(train)

ax=sns.countplot(x='Target', data=train, order=[4,3,2,1])
plt.title('Distribution of Target')
plt.xlabel('Target')
labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
ax.set_xticklabels(labels, rotation=0)

#Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Proportion [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))


# There are 2973 heads of households in the training set. They represent about 32.1% of observations in the dataset. 

# In[29]:


ax=sns.countplot(x='parentesco1', data=train, order=[1,0])
plt.title('Distribution of Head of Household')
plt.xlabel('Head of Household')
labels = ['Head of Household', 'Not Head of Household']
ax.set_xticklabels(labels, rotation=0)


#Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Proportion [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))


# Despite a high level of missingness in the variable v2a1, the boxplot below shows an unsurpring pattern. The montly rent for "Non-Vulnerable" houses is highest and lowest for "Extreme" households. 

# In[30]:


#monthly rent 
ax= sns.boxplot(x="Target", y="v2a1", data=train, order=[4,3,2,1])
ax.set(yscale="log")
ax.set_ylabel("Monthly Rent")
labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
ax.set_xticklabels(labels, rotation=0);


# Similarly, the number of years of education of individuals seems to vary between target levels. The median individual from "Non-vulnerable" households have the more education than the median invidual from "Extreme" households have the least. 

# In[31]:


ax= sns.boxplot(x="Target", y="escolari", data=train, order=[4,3,2,1])
ax.set_ylabel("Years of Education")
labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
ax.set_xticklabels(labels, rotation=0);


# The number of children per household seems to vary by target level. The median number of children of an individual is higher for individuals with "Extreme" houses than the number of children in other target levels. 

# In[32]:


# number of children
ax= sns.boxplot(x="Target", y="hogar_nin", data=train, order=[4,3,2,1])
ax.set_ylabel("Number of Children in Household")
labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
ax.set_xticklabels(labels, rotation=0);


# Age seems to vary by target level. The median age of "Non-vulnerable" observations is higher that those observations that indicated other levels of house quality. As the status becomes more serious, the median age decreases respectively. 

# In[33]:


#age
ax= sns.boxplot(x="Target", y="age", data=train, order=[4,3,2,1])
ax.set_ylabel("Age")
labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
ax.set_xticklabels(labels, rotation=0);


# The percent of observations indicating overcrowding is higher as the level of vulnerability increases. Observations with "Extreme" class labels indicated higher levels of overcrowding than any other target level. 

# In[34]:


#overcrowding
p_table = pd.pivot_table(train, index='Target', columns='hacdor', aggfunc='size')
p_table = p_table.div(p_table.sum(axis=1), axis=0)
p_table = p_table.loc[[4,3,2,1]]
p_table.plot.bar(stacked=True)

labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
plt.xticks([0,1,2,3],labels, rotation=0);


# Most observations in the training data come from the Central Region. 58.8% of observations are from that region and 41.2% were from other regions. 

# In[35]:


#region from all houses what percentage are in central region 
ncount = len(train)

ax=sns.countplot(x='lugar1', data=train, order=[1,0])
plt.title('Distribution of Households Region Central')
plt.xlabel('')
labels = ['Central Region', 'Non-Central Region']
ax.set_xticklabels(labels, rotation=0)


#Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Proportion [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))


# In[36]:


#region from all houses of each target what percentage are in central region
p_table = pd.pivot_table(train, index='Target', columns='lugar1', aggfunc='size')
p_table = p_table.div(p_table.sum(axis=1), axis=0)
p_table = p_table.loc[[4,3,2,1]]
p_table.plot.bar(stacked=True)

labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
plt.xticks([0,1,2,3],labels, rotation=0);


# In[37]:


#from houses in central region, what percentage does each target represent
central=train.loc[train['lugar1'] == 1]


ncount = len(central)

ax=sns.countplot(x='Target', data=central, order=[4,3,2,1])
plt.title('Distribution of Target in Central Region')
plt.xlabel('Target')
labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
ax.set_xticklabels(labels, rotation=0)

#Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Proportion [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10));


# In[38]:


#region from all houses what percentage are in central chorotega 
ncount = len(train)

ax=sns.countplot(x='lugar2', data=train, order=[1,0])
plt.title('Distribution of Households Region Chorotega')
plt.xlabel('')
labels = ['Chorotega Region', 'Non-Chorotega Region']
ax.set_xticklabels(labels, rotation=0)


#Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Proportion [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))


# In[39]:


#from houses in chorotega region, what percentage does each target represent
chorotega=train.loc[train['lugar2'] == 1]

ncount = len(chorotega)

ax=sns.countplot(x='Target', data=chorotega, order=[4,3,2,1])
plt.title('Distribution of Target in Central Chorotega')
plt.xlabel('Target')
labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
ax.set_xticklabels(labels, rotation=0)

#Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Proportion [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))


# In[40]:


#region from all houses what percentage are in pacifico central 
ncount = len(train)

ax=sns.countplot(x='lugar3', data=train, order=[1,0])
plt.title('Distribution of Households Region Pacifico Central')
plt.xlabel('')
labels = ['PC Region', 'Non-PC Region']
ax.set_xticklabels(labels, rotation=0)


#Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Proportion [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))


# In[41]:


#from houses in PC region, what percentage does each target represent
PC=train.loc[train['lugar3'] == 1]

ncount = len(PC)

ax=sns.countplot(x='Target', data=PC, order=[4,3,2,1])
plt.title('Distribution of Target in PC')
plt.xlabel('Target')
labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
ax.set_xticklabels(labels, rotation=0)

#Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Proportion [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))


# In[42]:


#from houses in Brunca region, what percentage does each target represent
BR=train.loc[train['lugar4'] == 1]

ncount = len(BR)

ax=sns.countplot(x='Target', data=BR, order=[4,3,2,1])
plt.title('Distribution of Target in BR')
plt.xlabel('Target')
labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
ax.set_xticklabels(labels, rotation=0)

#Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Proportion [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))


# In[43]:


#from houses in Brunca region, what percentage does each target represent
huetarAt=train.loc[train['lugar4'] == 1]

ncount = len(huetarAt)

ax=sns.countplot(x='Target', data=huetarAt, order=[4,3,2,1])
plt.title('Distribution of Target in huetarAt')
plt.xlabel('Target')
labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
ax.set_xticklabels(labels, rotation=0)

#Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Proportion [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))


# In[44]:


#from houses in Huetar region, what percentage does each target represent
huetarNort=train.loc[train['lugar5'] == 1]

ncount = len(huetarNort)

ax=sns.countplot(x='Target', data=huetarNort, order=[4,3,2,1])
plt.title('Distribution of Target in huetarNort')
plt.xlabel('Target')
labels = ['Non-vulnerable', 'Vulnerable', 'Moderate', 'Extreme']
ax.set_xticklabels(labels, rotation=0)

#Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Proportion [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))


# Correlation between some of the numeric variables in the dataset

# In[45]:


numeric_vars = train.loc[:, ['v2a1', 'rooms', 'r4h1', 'r4h2','r4h3', 'r4m3', 'r4m1',
                         'r4m2', 'r4t1', 'r4t2', 'tamviv', 'escolari', 'rez_esc',
                         'hhsize', 'hogar_nin', 'hogar_adul', 'hogar_mayor',
                         'hogar_total', 'dependency', 'v18q1','edjefe', 'edjefa',
                         'meaneduc', 'bedrooms','overcrowding','age']]

corr=numeric_vars.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False);


# ### Base Random Forest

# #### Base Random Forest: Traning

# In[4]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

rf_base=RandomForestClassifier(random_state=42)
rf_base.get_params()


# In[5]:


#get drop predictors of ID and those with missing data 
train_x=train.drop(['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 
                       'SQBmeaned', 'Target', 'idhogar',  
                    'Id', 'dependency','edjefe','edjefa'], axis=1)

#get target variable
train_y=train[["Target"]]

#fit base random forest
rf_base.fit(train_x, train_y.values.ravel());

#confusion matrix
metrics.confusion_matrix(train_y, rf_base.predict(train_x))


# #### Base Random Forest: Test

# In[6]:


#read test data
test = pd.read_csv("test.csv").set_index('Id')

#drop predictors of ID and those with missing data 
#get drop predictors of ID and those with missing data 
test_x=test.drop(['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 
                       'SQBmeaned', 'idhogar',  
                 'dependency','edjefe','edjefa'], axis=1)

Target=pd.DataFrame(rf_base.predict(test_x))

output_rf_base=pd.concat([test[["Id"]], Target], axis=1)
output_rf_base.columns = ["Id", "Target"]
output_rf_base.to_csv("output_rf_base.csv", index=False)


# ### Random Forest with SMOTETomek Oversampling to match majority class

# #### Resample data

# In[7]:


from collections import Counter
from imblearn.combine import SMOTETomek


smt = SMOTETomek(random_state=42)
train_bal_x, train_bal_y = smt.fit_sample(train_x, train_y.values.ravel())
print('Resampled dataset shape {}'.format(Counter(train_bal_y)))


# #### Train on Balanced Random Forest

# In[50]:


rf_bal=RandomForestClassifier(random_state=42)

#fit base random forest
rf_bal.fit(train_bal_x, train_bal_y);

#confusion matrix
metrics.confusion_matrix(train_bal_y, rf_bal.predict(train_bal_x))


# #### Test Balanced Random Forest

# In[51]:


Target=pd.DataFrame(rf_bal.predict(test_x))

output_rf_bal=pd.concat([test[["Id"]], Target], axis=1)
output_rf_bal.columns = ["Id", "Target"]
output_rf_bal.to_csv("output_rf_bal.csv", index=False)


# ### Random Forest with SMOTETomek K-fold cross validation 

# In[54]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

# Minimum number of samples required to split a node
min_samples_split = [10,20,30,40]

# Minimum number of samples required at each leaf node
min_samples_leaf = [10, 20, 30]
# Method of selecting samples for training each tree

bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf=RandomForestClassifier(random_state=42)


rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid,
                               n_iter = 100, 
                               cv = 10, 
                               verbose=2, 
                               random_state=42,
                               n_jobs = -1)

rf_random.fit(train_bal_x, train_bal_y)


# In[55]:


rf_random.best_params_


# In[61]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap': [False],
    'max_depth': [55, 60, 65],
    'min_samples_leaf': [8, 10, 15],
    'min_samples_split': [8, 10, 15],
    'n_estimators': [400, 450, 500]
}
# Create a based model
rf = RandomForestClassifier(random_state=42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 2)

grid_search.fit(train_bal_x, train_bal_y)


# In[62]:


grid_search.best_params_


# In[64]:


Target=pd.DataFrame(grid_search.best_estimator_.predict(test_x))

    
output_rf_grid_best=pd.concat([test[["Id"]], Target], axis=1)
output_rf_grid_best.columns = ["Id", "Target"]
output_rf_grid_best.to_csv("output_rf_grid_best.csv", index=False)


# #### Combine Random Undersampling and Oversampling

# In[28]:


from sklearn.utils import resample

class_1=train.loc[train["Target"]==1]
class_2=train.loc[train["Target"]==2]
class_3=train.loc[train["Target"]==3]
class_4=train.loc[train["Target"]==4]


class_1_over = resample(class_1, 
                        replace=True,    # sample without replacement
                        n_samples=3000,     # to match minority class
                        random_state=123) 

class_2_over = resample(class_2, 
                        replace=True,    # sample without replacement
                        n_samples=3000,     # to match minority class
                        random_state=123) 

class_3_over = resample(class_3, 
                        replace=True,    # sample without replacement
                        n_samples=3000,     # to match minority class
                        random_state=123) 

train_rand_bal = pd.concat([class_1_over, class_2_over,class_3_over,class_4])
train_rand_bal=train_rand_bal.sample(frac=1)


rf = RandomForestClassifier(random_state=42
                           bootstrap=False,
                            max_depth=55,
                            min_samples_leaf': 8,
                            min_samples_split': 8,
                            n_estimators': 500)


# ####  Adaboost with SMOTETOMEK

# In[18]:


from sklearn.ensemble import AdaBoostClassifier

adboost = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1,
                         random_state=0)

model = adboost.fit(train_bal_x, train_bal_y)

Target=pd.DataFrame(model.predict(test_x))
    
output_adboost=pd.concat([test[["Id"]], Target], axis=1)
output_adboost.columns = ["Id", "Target"]
output_adboost.to_csv("output_output_adboost_base.csv", index=False)


# In[23]:




