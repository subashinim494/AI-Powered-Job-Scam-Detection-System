import  numpy as np
import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("fake_job_postings.csv")
df.head()
df.shape
df.info()
df.nunique()
df.isnull().sum()
df['fraudulent'].value_counts()

df['location'].fillna('Unknown', inplace=True)
df['department'].fillna('Unknown', inplace=True)
df['salary_range'].fillna('Not Specified', inplace=True)
df['employment_type'].fillna('Not Specified', inplace=True)
df['required_experience'].fillna('Not Specified', inplace=True)
df['required_education'].fillna('Not Specified', inplace=True)
df['industry'].fillna('Not Specified', inplace=True)
df['function'].fillna('Not Specified', inplace=True)

df.isnull().sum()

na_columns = ['company_profile', 'description', 'requirements', 'benefits']
df[na_columns] = df[na_columns].fillna('Missing')

df.isna().sum()

numerical_columns = []
text_columns = []

for col in df.columns:
    if df[col].dtype == 'object':
        text_columns.append(col)
    else:
        numerical_columns.append(col)
print("Numerical Columns: ", numerical_columns)

df[numerical_columns].describe()
print("Text Columns: ", text_columns)
df.head()
df['fraudulent'].value_counts()
sns.countplot(x='fraudulent', data=df)
plt.title('Distribution of Fraudulent Job Postings')
plt.show()
sns.countplot(data = df, x = 'employment_type')
plt.title('Distribution of Employment Type')
df.columns
plt.figure(figsize=(15 ,6))
sns.barplot(data = df, x = 'required_experience', y = 'fraudulent')
plt.figure(figsize=(20, 10))
sns.barplot(data=df, x='required_education', y='fraudulent', estimator=sum)
plt.title('Fraudulent Postings by Required Education')
plt.xlabel('Required Education')
plt.ylabel('Sum of Fraudulent Postings')
plt.xticks(rotation = 90)
plt.show()
df.columns
fraudulent_summary = df.groupby('function')['fraudulent'].sum().reset_index()

plt.figure(figsize=(25, 8))
sns.lineplot(data=fraudulent_summary, x='function', y='fraudulent', marker='o')
plt.title('Fraudulent Postings by Function')
plt.xlabel('Function')
plt.ylabel('Sum of Fraudulent Postings')
plt.xticks(rotation=45)
plt.grid(True) 
plt.show()
plt.figure(figsize=(13,5))
country = df['required_education'].value_counts().nlargest(5).index.tolist()
ax=sns.countplot(data=df, x='required_education', order=country, palette='rainbow')
ax.set_ylabel('No. of Jobs')