import streamlit as st
import pandas as pd
import joblib

df = pd.read_csv('app_building_df.csv')

st.header('Sao Carlos House Price Prediction', divider='rainbow')
# get dropdown for Category
cat = st.selectbox('What House Category do you want?', df['Category'].unique())

# get dropdown for Subcategory
sub_cat = st.selectbox('What is the Subcategory?', df['Subcategory'].unique())

# Input built up
built_area = st.slider('Built up Area', 0.0, max(df["Built_up_area"]), 1.0)

# Insert Total area
total_area = st.slider('Built up Area', 0.0, max(df["Total_area"]), 1.0)

# Insert number of Bedrooms
bedrooms = st.number_input("Number of bedrooms", min_value=0, value=0)

# Insert number of bathrooms
bathrooms = st.number_input("Number of bathrooms", min_value=0, value=0)


# Insert number of Garages
garages = st.number_input("Number of garages", min_value=0, value=0)

# Insert Total number of rooms
total_rooms = st.number_input("Total Number of Rooms", min_value=0, value=0)

# Insert prop_char_95
prop_char_95 = st.number_input("Prop Char 95?")

# Insert Adress
# df['District'] + df['Street_name']+df['Condominium']
address = st.text_input("Address of the Building?\n(District + Street name + Condominium)")

cols = ['Category', 'Subcategory', 'Built_up_area', 'Total_area', 'Bedroom',
       'Bathroom', 'Garages', 'totalRooms', 'prop_char_95', 'Address']

lst = [cat, sub_cat, built_area, total_area, bedrooms,
        bathrooms, garages, total_rooms, prop_char_95, address]

result = ''
if st.button('Get House Price Prediction'):
    # Loadng the Transformer & the Model

    # Load the combined_pipeline from the pickle file
    loaded_combined_pipeline = joblib.load('combined_pipeline.pkl')

    # Load the best model from the 'best_model.pkl' file
    loaded_best_model = joblib.load('best_model.pkl')

    new_df = pd.DataFrame([lst], columns=cols)

    # Transform the Categorical Features
    new_x = loaded_combined_pipeline.transform(new_df)
    # make the predictions
    pred_price = loaded_best_model.predict(new_x)

    # Write the predictions
    result = pred_price[0]
st.success(f'House Price is: **{result}**')
# st.success(f'House Price is: **{result:,}**')



