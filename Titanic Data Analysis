import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


gender_df = pd.read_csv('gender_submission.csv')
test_df = pd.read_csv('test.csv')
titanic_df = pd.read_csv('Titanic.csv')


# Display Top 5 Rows of The Dataset
print(titanic_df.head())


# Check the Last 3 Rows of The Dataset
print(titanic_df.tail(3))


# Find Shape of Our Dataset (Number of Rows & Number of Columns)
print(f'Number of rows: {titanic_df.shape[0]}\nNumber of Columns: {titanic_df.shape[1]}')


# Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory
print(titanic_df.info())


# Get Overall Statistics About The Dataframe
print(titanic_df.describe())


# Data Filtering
print(titanic_df.columns)

print(titanic_df[['Name','Age','Sex']])

print(titanic_df[titanic_df['Sex']=='male'].count())

print(titanic_df[titanic_df['Survived']==1])


# Check Null Values In The Dataset
print(titanic_df.isnull().sum())

print(titanic_df.isna().sum())

sns.heatmap(titanic_df.isnull())
plt.show()

missing_per = titanic_df.isnull().sum()*100/len(titanic_df)
print(missing_per)


# Drop the Column
titanic_df.drop('Cabin', axis=1, inplace=True)
print(titanic_df.info())

missing_per = titanic_df.isnull().sum() * 100 / len(titanic_df)
print(missing_per)


# Handle Missing Values
print(titanic_df.columns)

print(titanic_df['Embarked'].mode())
titanic_df['Embarked'].fillna('S', inplace = True) #handling categorical missing values

titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace = True)
print(titanic_df.isnull().sum())


# Categorical Data Encoding
print(titanic_df['Sex'].unique())

titanic_df['Gender'] = titanic_df['Sex'].map({'male':1,'female':0})
print(titanic_df)

titanic_df.insert(5, 'Gen', titanic_df['Sex'].map({'male':1,'female':0}))
print(titanic_df.info())

print(titanic_df['Embarked'].unique())

print(pd.get_dummies(titanic_df, columns=['Embarked']))
print(titanic_df)

print(pd.get_dummies(titanic_df, columns=['Embarked'], drop_first=True))
print(titanic_df)


# What is Univariate Analysis? How Many People Survived And How Many Died?
# How Many Passengers Were In 1st, 2nd & 3rd Class?
# Number of Male And Female Passengers
print(titanic_df['Survived'].value_counts())

sns.countplot(x=titanic_df['Survived'])
plt.show()

sns.countplot(x=titanic_df['Pclass'])
plt.show()

sns.histplot(titanic_df['Age'])
plt.show()

print(sns.boxplot(titanic_df['Age'], orient='v'))
plt.show()


# Bivariate Analysis How Has Better Chance of Survival Male or Female?Which Passenger Class Has more Chance of Survival?
print(sns.countplot(x=titanic_df['Survived'] , hue=titanic_df['Sex']))
plt.show()

print(sns.countplot(x=titanic_df['Pclass'], hue=titanic_df['Survived']))
plt.show()

sns.barplot(x=titanic_df['Sex'], y=titanic_df['Survived'])
plt.show()

sns.barplot(x='Pclass', y='Survived', data=titanic_df)
plt.show()


# Feature Engineering
print(titanic_df.columns)
titanic_df['Family'] = titanic_df['SibSp']+titanic_df['Parch']
titanic_df['Fare_PP'] = titanic_df['Fare']/(titanic_df['Family']+1)
print(titanic_df.head(1))
