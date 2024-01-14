# demand.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Convert travel_date and travel_time to datetime
    df['datetime'] = pd.to_datetime(df['travel_date'] + ' ' + df['travel_time'])

    # Extract features
    features = df[['travel_from', 'datetime', 'car_type']]

    # Convert datetime to Unix timestamp
    features['datetime'] = (df['datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    # Encode categorical variables
    features = pd.get_dummies(features, columns=['travel_from', 'car_type'], drop_first=True)

    # Target variable
    target = df['max_capacity']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)

    return df, model

def show_demand_analysis(df):
    st.subheader('Demand Prediction Analysis')

    # Sidebar menu for demand navigation
    demand_page = st.sidebar.selectbox("Choose a demand page", ['Home', 'Number of Tickets Distribution', 'Rides from Each Location', 'Car Types Count', 'Max Capacity Count', 'Datetime vs Max Capacity'])

    if demand_page == 'Home':
        st.write("Welcome to the Demand Prediction Analysis!")
        st.write("Use the sidebar to navigate between different demand analysis pages.")

    elif demand_page == 'Number of Tickets Distribution':
        # Histogram for number_of_ticket distribution
        st.subheader('Distribution of Number of Tickets')
        st.write("This histogram displays the distribution of the number of tickets sold.")

        bins = st.slider("Select number of bins", 5, 50, 15)
        fig, ax = plt.subplots(figsize=(12, 6))

        # Modify this line to check if 'number_of_ticket' is in the DataFrame
        if 'number_of_ticket' in df.columns:
            sns.histplot(df['number_of_ticket'], bins=bins, kde=True, ax=ax)
        else:
            st.write("Error: 'number_of_ticket' column not found in the DataFrame.")

        st.pyplot(fig)

    elif demand_page == 'Rides from Each Location':
        # Barplot for travel_from counts
        st.subheader('Count of Rides from Each Location')
        st.write("This barplot shows the count of rides originating from each location.")
        hue_col = st.selectbox("Select a hue for the count plot", ['car_type', 'travel_to', None])
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='travel_from', data=df, ax=ax, hue=hue_col)
        st.pyplot(fig)

    elif demand_page == 'Car Types Count':
        # Countplot for car_type
        st.subheader('Count of Car Types')
        st.write("This countplot visualizes the distribution of different car types in the dataset.")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='car_type', data=df, ax=ax)
        st.pyplot(fig)

    elif demand_page == 'Max Capacity Count':
        # Countplot for max_capacity counts
        st.subheader('Count of Max Capacity')
        st.write("This countplot displays the distribution of maximum capacities for the vehicles.")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='max_capacity', data=df, ax=ax)
        st.pyplot(fig)

    elif demand_page == 'Datetime vs Max Capacity':
        # Scatterplot
        st.subheader('Scatter Plot: Datetime vs Max Capacity')
        st.write("This scatter plot shows the relationship between the datetime of a ride and its maximum capacity.")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x='datetime', y='max_capacity', data=df, ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    st.set_page_config(page_title="Demand Prediction App", page_icon=":chart_with_upwards_trend:")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file for Demand Prediction", type=["csv"])

    # Load data if file is uploaded
    if uploaded_file:
        df, demand_model = load_data(uploaded_file)
        show_demand_analysis(df)
