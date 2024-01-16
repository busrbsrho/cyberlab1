#!/usr/bin/env python
# coding: utf-8

# # Class Hands on Lab

# # I recommend to install python Anaconda
# ## The following is a list of installs if you want to use your own anaconda.
# ## How to run each cell? ==> shift +Enter

# In[4]:


#!pip install matplotlib
#!pip install pandas
#!pip install seaborn
#!pip install scipy
#!pip install sklearn
get_ipython().system('pip install -U scikit-learn')


# In[5]:


# Imports
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings


# # try me:

# In[3]:


y = 2+5
print(y)


# In[4]:


print("this is  Y:",y)


# To read dataframe we will use Pandas.  Try pd. (press Tab -> will show you options)
# Docs: https://pandas.pydata.org/docs/

# In[5]:


# how to read a CSV file:
pd.read_csv(<path>,sep=",")


# ## Read the data

# In[12]:


# file path - this for linux windows you will need "//"
f_path = "conn_attack.csv"

'''
record ID - The unique identifier for each connection record.
duration_  This feature denotes the number of seconds (rounded) of the connection. For example, a connection for 0.17s or 0.3s would be indicated with a “0” in this field.
src_bytes This field represents the number of data bytes transferred from the source to the destination (i.e., the amount of out-going bytes from the host).
dst_bytes This fea
ture represents the number of data bytes transferred from the destination to the source (i.e., the amount of bytes received by the host).
'''
df = pd.read_csv(f_path,names=["record ID","duration_", "src_bytes","dst_bytes"], header=None)


# In[13]:


# we could have nan values in the dataset (issue in the data) lets dropna()
df.head()


# Get the head of the data

# In[14]:


# if you will add "." and later TAB you will see the different options
df.


# In[15]:


df.head()


# # # Pandas DataFrame

# ### A.  Handling missing data - detect if we have this case:

# In[16]:


# determin the missing data precentage
df.apply(lambda x: sum(x.isna()) / len(df))


# ### B. Remove na/nan:

# ##The Pandas function dropna() drops rows or columns (depending on the parameter you choose) that contain missing values. This function takes the axis parameter which you set as 0 to drop rows, and 1 to drop columns.
# 
# Please note that:
# 
# The alternative function is fillna() . This function will replace missing values with the value of your choice. You can replace with a fixed value such as 0, or you can use a calculation such as the mean. You can also apply different values to different columns by passing a dictionary of values per column.

# In[ ]:


df.shape


# In[ ]:


# check for nan values
df.isnull().any().any() # check if we have nan values.


# #### Solution one: Drop nan

# In[ ]:


df = df.dropna()


# In[ ]:


df.shape


# #### imputation - replace nan with: mode() - most common value

# The below code fills any missing values with the mode for that column. We used fil when we have features that do not have all placement in all cases.

# In[ ]:


df_numeric = df.apply(lambda x: x.fillna(x.mode()),axis=0)


# #### C. Selecting subsets from our data

# The loc method selects rows based on the index label. Let’s walk through a quick example.

# In[33]:


df.head()


# In[10]:


number_range = range(0,200)
print(number_range)


# In[34]:


subset_loc = df.loc[number_range]
subset_loc#.head()


# In[35]:


subset_loc.shape


# The iloc method select rows by the index position. This might be used, for example, if the user does not know the index or if the index is not numeric.
# 
# 
# Similar to loc

# In[28]:


subset_iloc = df.iloc[[0, 1, 2]]
subset_iloc.head()


# In[18]:


df.src_bytes


# In[20]:


df["src_bytes"]


# #### D. DataFrame "Where" The SQL alternative for searching

# In[29]:


df.where(df['src_bytes'] > 240).dropna()


# In[23]:


# or /use this:
df[df['src_bytes'] > 240]


# In[21]:


#or more complex:
df[(df['src_bytes']> 240) & (df['dst_bytes']> 1000)]


# In[22]:


ran_df = df[(df['src_bytes']> 240) & (df['dst_bytes']> 1000)]


# In[23]:


ran_df.shape


# #### E. Describe

# In[24]:


df.describe()


# #### F. Dataset statistics calculations

# In[26]:


#Mean
df['dst_bytes'].mean()


# In[25]:


df.dst_bytes.median()


# In[7]:


#Median of two seperate columns
df[["src_bytes", "dst_bytes"]].median()


# In[8]:


#Instead of the predefined statistics, specific combinations can be calculated
df.agg(
    {
        "duration_": ["min", "max", "median", "skew"],
        "dst_bytes": ["min", "max", "median", "mean"],
    }
)


# #### G. Grouping

# In[29]:


df[["src_bytes", "dst_bytes"]].groupby("src_bytes").std()


# In[36]:


df.groupby(["src_bytes", "dst_bytes"])["duration_"].mean()


# ## Data exploration
# 
# ### Explore the data, understand the featues, statistics visualize the inputs
# #### please try the following tools and extend them: this is for a soft start :) Explain how you explore the data. Why this is important? Please note this are only some examples
# 

# ### Skewness:  is a statistical measure that describes the asymmetry of a distribution around its mean, indicating whether the distribution tails lean left or right. 
# ### Kurtosis:  on the other hand, measures the 'tailedness' of a distribution, indicating whether the data have heavy or light tails compared to a normal distribution.

# ![868px-Asymmetric_Distribution_with_Zero_Skewness.jpeg](attachment:868px-Asymmetric_Distribution_with_Zero_Skewness.jpeg)

# ![868px-Relationship_between_mean_and_median_under_different_skewness.png](attachment:868px-Relationship_between_mean_and_median_under_different_skewness.png)

# In[28]:


df['src_bytes'].mean()


# In[32]:


#histogram
sns.distplot(df['src_bytes'])


# In[33]:


'''
Deviate from the normal distribution.
Have appreciable positive skewness.
Show peakedness.
'''
#skewness and kurtosis
print("Skewness: %f" % df['src_bytes'].skew())
print("Kurtosis: %f" % df['src_bytes'].kurt()) # positive show long tail , negative light tail


# In[36]:


sns.distplot(df['dst_bytes'])
print("Skewness: %f" % df['dst_bytes'].skew())
print("Kurtosis: %f" % df['dst_bytes'].kurt())


# In[33]:


#Relationship with numerical variables
var = 'dst_bytes'
data = pd.concat([df['src_bytes'], df[var]], axis=1)
data.plot.scatter(x=var, y='src_bytes', ylim=(0,100000)); # do not `assume here any thing this is just examples


# In[34]:


#scatter plot totalbsmtsf/saleprice
var = 'duration_'
data = pd.concat([df['src_bytes'], df[var]], axis=1)
data.plot.scatter(x=var, y='src_bytes', ylim=(0,100000));


# 
# ### Correlation is a statistical term which in common usage refers to how close two variables are to having a linear relationship with each other.
# ### Negative -  different directions
# ### zero -  no correlation
# ### one - identical

# In[35]:


df.corr() 


# In[36]:


# Increase the size of the heatmap.
plt.figure(figsize=(16, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12});


# In[37]:


plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df.corr()[['src_bytes']].sort_values(by='src_bytes', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with src_bytes', fontdict={'fontsize':18});


# # Introduction to ML in python

# I will take here a supervided dataset. The idea is to demonstrate you some concepts

# In[10]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
# Load Dataset
X, y = datasets.load_iris(return_X_y=True) # note this is a known dataset with dedicated loader. In your work you can use a custom one.
X.shape, y.shape


# ### Dataset:
# Train and Validation are used in the training while Test is left out for model verification

# ![image.png](attachment:image.png)

# # Introduction to Cross Validation

# In machine learning we need to estimate the performance of a model before we put it into production. 
# 
# While we could just evaluate our model's performance on the same data that we used to fit its parameters.
# doing so will give us unreliable assessments of our model's ability to generalize to unseen data.
# 
# source: https://mlu-explain.github.io/cross-validation/

# ### Leave-One-Out Cross Validation (LOOCV)

# ![image.png](attachment:image.png)

# ### K-Fold Cross-Validation

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# Let's split the data 80:20

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

print("Train:",X_train.shape, y_train.shape)

print("Test", X_test.shape, y_test.shape)

# Validation can be taken from the Train:
# Further split the training set into 75% training and 25% validation
# This results in 60% training, 20% validation, and 20% testing of the original dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=2)

X_train.shape, y_train.shape

X_test.shape, y_test.shape
'''
When evaluating different settings (“hyperparameters”) for estimators, such as the C 
setting that must be manually set for an SVM, there is still a risk of overfitting
on the test set because the parameters can be tweaked until the estimator performs optimally. 
You can optimize "C" with GridSearch.
'''

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print('the accuracy here is :',clf.score(X_test, y_test))
#let's predict the data
y_predicted = clf.predict(X_test)
# lets review the labels
labels = np.unique(y)
print("unique labels",labels)
# The confusion matrix is:
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predicted, labels=labels)


# Here we have a problem:
#     
#     Our data is trained with non optimized parameters.
# 
# How can we solve it?:
#      
#      cross-validation  + k-Fold
#      
#      The k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”:
# 
#     A model is trained using  of the folds as training data;
# 
#     the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).
#     

# In[17]:


from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print("we are getting in scores the result of each iterration. Len:",len(scores), "values:", scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("The Max set was: ", scores.max())
print("NOTE: the result here are different! why? - OVERFITING")


# Now let's see how can we use data normalization with this example

# In[27]:


from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
print("Let's see if the training size effect us ? change the test size from 0.2 to 0.4")
#standardScaler normalization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
print("result score: ",clf.score(X_test_transformed, y_test))


#changing:
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)
print("changed")
#standardScaler normalization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
print("result score: ",clf.score(X_test_transformed, y_test))
clf.predict(X_test_transformed)
confusion_matrix(y_test, y_predicted, labels=labels)


# ### Data Normalization

# In[ ]:


Standardize features by removing the mean and scaling to unit variance.

The standard score of a sample x is calculated as:

z = (x - u) / std


# In[28]:


from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data))
print("The mean is : ", scaler.mean_)
print(scaler.transform(data), " the mean: ",scaler.transform(data).mean(),"STD: ", scaler.transform(data).std() )
print(scaler.transform([[2, 2]]))


# Lets write it with piplines:

# In[29]:


from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
cross_val_score(clf, X, y, cv=10) #note -we can use other CV methods see sklearn.


# Let's try a different pipline approach

# In[30]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
classifier_names = ["Logistic Regression", "KNN", "Random Forest","SVM"]

classifiers = [LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(), LinearSVC()]

zipped_clf = zip(classifier_names,classifiers)


# In[31]:


def classifier(classifier, t_train, c_train, t_test, c_test):
    result = []
    for n,c in classifier:
        checker_pipeline = Pipeline([
            ('standardize', StandardScaler()),                         
            ('classifier', c)
        ])
        print("Validation result for {}".format(n))
        print(c)
        clf_acc = fit_classifier(checker_pipeline, t_train, c_train, t_test,c_test)
        result.append((n,clf_acc))
    return result


# In[32]:


def fit_classifier(pipeline, x_train, y_train, x_test, y_test):
    model_fit = pipeline.fit(x_train, y_train)
    y_pred = model_fit.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    return accuracy


# In[33]:


result = classifier(zipped_clf, X_train, y_train, X_test, y_test)


# ### Measuring metrics:
# #### Accuracy:  shows how often a classification ML model is correct overall. 
# #### Precision shows how often an ML model is correct when predicting the target class.
# #### Recall shows whether an ML model can find all objects of the target class. 
# 
# Figure are taken from here: https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall#:~:text=TL%3BDR,objects%20of%20the%20target%20class.

# ![image.png](attachment:image.png)
# 

# ![image.png](attachment:image.png)

# In our case, 52 out of 60 predictions (labeled with a green “tick” sign) were correct. Meaning the model accuracy is 87%. 

# ![image-2.png](attachment:image-2.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# Please note in this example we didn't optimize the classifiers.

# # :) Back to the task :)

# TO SUBMIT: Summary of data exploration:
# 
#     Did this help you?
#     A .What did you find? [5 points]
#     B. What the data exploration tell you about the data and the possible attacks?[10 points]

# # Supervised Machine Learning

# Load the original dataset conn250K.csv and the label dataset:conn250K_anomaly.csv

# In[14]:


# file path - this for linux windows you will need "//"
f_path = "conn_attack.csv"

'''
record ID - The unique identifier for each connection record.
duration_  This feature denotes the number of seconds (rounded) of the connection. For example, a connection for 0.17s or 0.3s would be indicated with a “0” in this field.
src_bytes This field represents the number of data bytes transferred from the source to the destination (i.e., the amount of out-going bytes from the host).
dst_bytes This fea
ture represents the number of data bytes transferred from the destination to the source (i.e., the amount of bytes received by the host).
'''
df = pd.read_csv(f_path,names=["record ID","duration_", "src_bytes","dst_bytes"], header=None)


# In[72]:


df.shape
df


# In[17]:


labels = pd.read_csv("conn_attack_anomaly_labels.csv",names=["record ID","label"], header=None)


# In[20]:


labels


# In[19]:


df_label = labels.label


# In[29]:


df_label.values


# In[30]:


labels.shape


# In[31]:


np.unique(labels.label)


# 1. Create a supervised machine learning (use the labels). [20 points]
# 

# In[29]:


#|df



# In[52]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

features = ["duration_","src_bytes","dst_bytes"]
X = df[features]
y = df_label.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Train:",X_train.shape, y_train.shape)
print("Test", X_test.shape, y_test.shape)



clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
print('the accuracy here is :',clf.score(X_test, y_test))


#print(clf.predict(X))


# In[41]:


X.head()


# In[33]:


y


# In[40]:


Short Video About Random Forest:
    https://www.youtube.com/watch?v=v6VJ2RO66Ag&ab_channel=NormalizedNerd


# In[75]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
features = ["duration_","src_bytes","dst_bytes"]
df_label = labels.label
X = df[features]
y = df_label.values
clf = RandomForestClassifier(max_depth=2, random_state=0)
#spliting the data to run the randomforest algorithm
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
clf.fit(X_train, y_train)
#print(clf.predict(X_test))
print('the test accuracy after splittin 20% here is :',clf.score(X_test, y_test))
print('the train accuracy after splittin 20% here is :',clf.score(X_train, y_train))



#training the data at least i think so
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=0)
clf.fit(X_train, y_train)
#print(clf.predict(X_test))
print('the test accuracy after splitting 40% here is :',clf.score(X_test, y_test))
print('the train accuracy after splitting 40% here is :',clf.score(X_train, y_train))





#print(clf.predict(X_test))


# In[28]:


clf.predict(X) ==y


# In[82]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score

features = ["duration_", "src_bytes", "dst_bytes"]
#X = df[features]
#y = df_label.values
# create train and test sets -- TODO!!!
X_train, X_test, y_train, y_test = train_test_split(df[features], df_label, test_size=0.2)
#X = df y
#X.shape , y.shape
#rf = RandomForestClassifier(max_depth=2, random_state=0)

rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(X_train, y_train)
y_predicted = rf.predict(X_test)
accuracy = accuracy_score(y_predicted, y_test)

print("accuracy:", accuracy)
#confusion_matrix(y, y_predicted, labels=[0,1])


# In[83]:


from sklearn.metrics import classification_report

target_names = ['0', '1']
print(classification_report(y_predicted, y_test, target_names=target_names))


# 2. What machine learning algorithms did you used? Why did you used them? [5 points]
# 

# In[ ]:





# 4. Feature selection: which feature selection method did you used? [5 points]
# 

# In[ ]:





# 5. How did you measured the preformance of the machine learning [5 points]
# Note:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# 

# In[ ]:





# 6. plot AUC https://scikit-learn.org/1.0/modules/generated/sklearn.metrics.plot_roc_curve.html [5 points]
# 

# In[ ]:





# 7. Following your answer in section (2). Can you improve your results? [10]
# NOTE: https://scikit-learn.org/stable/modules/ensemble.html
# How much did you improved?

# In[ ]:





# # Unsupervised Learning (without labels)

# 8.Create Isolation forest algorithm compare your results to the supervised algorith. [20 points] measure your preformance usiong the labels
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

# In[ ]:





# 

# 9. Create DBSCAN algorithm [20 points] measure your preformance usiong the labels

# In[ ]:





# 10. Can you improve the unsupervised learning without using labels? Please write code [10 points] 

# In[ ]:





# # Create the final submission results

# What to submit:
# 
# 1.email ,full name and ID
# 
# 2. This notebook with all the answers. with diagrams
# 3. If you didnt manage to do some task please explain what did you tried to do.

# In[ ]:


1

