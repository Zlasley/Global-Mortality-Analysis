import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split














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
