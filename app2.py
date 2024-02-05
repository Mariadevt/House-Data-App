#Import the necessary libraries for the web app

import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



#Load the dataset
df = pd.read_excel("HouseData.xlsx")

#Add the title of our app
st.title("House Dataset App")

#Add an Image to our dataset
st.image("HouseDataset.png")

#Add the header

st.header("Dataset Concept", divider= "green")

st.write("""The house dataset encompasses a comprehensive collection of information pertaining to various residential properties. It includes essential features such as the number of bedrooms, bathrooms, square footage, and pricing details.
         Additionally, the dataset incorporates other relevant attributes like the presence of amenities, location-specific data, and the overall condition of the houses. This diverse dataset aims to provide a holistic view of the housing market, enabling users to analyze and understand the key factors influencing property values and buyer preferences.
         Through this dataset, one can explore the intricacies of the real estate landscape, facilitating data-driven insights for potential homebuyers, sellers, and industry analysts""")

# Display the header 

st.header("Explanatory Data Analysis(EDA)", divider="green")

#Use of the checkbox

if st.checkbox("Dataset Info"):
    st.write("Dataset Information", df.info())
    
if st.checkbox("Number of Rows"):
    st.write("Number of Rows", df.shape[0])

if st.checkbox("Number of Columns"):
    st.write("Number of Columns", df.shape[1])

if st.checkbox("Column Names"):
    st.write("Column Names", df.columns.tolist())
    
if st.checkbox("Data Types"):
    st.write("Data Types", df.dtypes)

if st.checkbox("Missing Values"):
    st.write("Missing Values", df.isnull().sum())
    
if st.checkbox("Statistical Summary"):
    st.write("Statistical Summary", df.describe())
    
#Visualisation Part of it

st.header("Visualization of the Dataset", divider="green")


#Line Graph 

#Dropdown to select columns for a Line Graph

linegraph_columns = st.multiselect("Line Graph", df.columns)


if linegraph_columns:
    st.line_chart(df[linegraph_columns].dropna())
    
else: 
    st.write("Select a column for Line Graph")

#BarGraph 

#Dropdown to select columns for a Bar Graph

bargraph_columns = st.multiselect("Bar Graph", df.columns)


if bargraph_columns:
    st.bar_chart(df[bargraph_columns].dropna())
    
else: 
    st.write("Select a column for Bar Graph")



#Encoding  using labelEncoder

hd = LabelEncoder()
df['date'] = hd.fit_transform(df['date'])
df['waterfront'] = hd.fit_transform(df['waterfront'])
df['view'] = hd.fit_transform(df['view'])
df['condition'] = hd.fit_transform(df['condition'])

# Encoding categorical columns using LabelEncoder
categorical_columns = ['date','waterfront','view','condition']
for column in categorical_columns:
    df[column] = hd.fit_transform(df[column])

#Use of the OneHotEncoder to encode the categorical features

ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(handle_unknown="ignore"),['date','waterfront','view','condition'])], remainder='passthrough')
X = df.iloc[:,: -1]
y = df.iloc[:,-1]
X_encoded = ct.fit_transform(df[['date','waterfront','view','condition']])

# Convert sparse matrix to dense array
X_encoded = X_encoded.toarray()

#split the data into training and testing 

X_train ,X_test, y_train,Y_test = train_test_split(X_encoded,y, test_size=0.2, random_state=0)

#Fit our regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# User input for independent variables
st.sidebar.header('Enter the values to be Predicted', divider="green")

# Create the input boxes for each feature
user_input = {}
for feature in df.drop("price", axis=1).columns:
    user_input[feature] = st.sidebar.text_input(f"Enter {feature}")

# Button to trigger the prediction
if st.sidebar.button('Predict'):
    # Create a dataframe for the user input
    user_input_df = pd.DataFrame([user_input])

    # Use the same ColumnTransformer to encode user input
    user_input_encoded = ct.transform(user_input_df)

    # Predict using the trained model
    y_pred = regressor.predict(user_input_encoded)

    # Display the predicted value
    st.write("Predicted Price:", y_pred[0])



    