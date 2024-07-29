import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Highest death rate%, what year, majority?  
# Can we predict rates within a certain degree? 

# read the CSV file with pandas
df = pd.read_csv(r"C:\Users\ALasl\Documents\GitHub\Mortality\archive")

print(df.head())

print( )

print(df.describe())

df_null = df.isnull().sum().sum()

print(df.columns.values.tolist())

print(f"how many null values are in the df: {df_null}")

print(f"How many rows: {df.len()}")

df = df.drop_duplicates(inplace = True)

# Set-up the one-hot encoder method
categorical_features = ['---']
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

# Set up our preprocessor/column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)])

# Add the classifier to the preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', DecisionTreeClassifier())])

train_test_split
