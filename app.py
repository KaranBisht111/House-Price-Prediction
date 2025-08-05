from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature columns
model = joblib.load('house_price_model.pkl')
columns = joblib.load('columns.pkl')

# Load the dataset for locations
df = pd.read_csv("bengaluru.csv")
df1 = df.drop(['area_type', 'society', 'availability', 'balcony'], axis=1)
df1 = df1.dropna()

# Function to clean the 'total_sqft' column
def clean_sqft(value):
    try:
        value = ''.join([i for i in value if i.isdigit() or i == '.'])
        return float(value)
    except:
        return None

df1['total_sqft'] = df1['total_sqft'].apply(clean_sqft)
df1 = df1.dropna(subset=['total_sqft'])

df1['bhk'] = df1['size'].apply(lambda x: int(x.split(' ')[0]))
df1.drop('size', axis=1, inplace=True)

df1['price'] = df1['price'] * 100000
df1['price_per_sqft'] = df1['price'] / df1['total_sqft']

df1['location'] = df1['location'].apply(lambda x: x.strip())
df1['location'] = df1['location'].apply(lambda x: 'other' if df1['location'].value_counts()[x] <= 10 else x)

locations = df1['location'].unique()

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    loc_index = np.where(columns == location)[0][0]
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    predicted_price = model.predict([x])[0]

    return render_template(
        'result.html',
        prediction_text=f'Predicted Price: ₹{predicted_price:,.2f} INR',
        location=location,
        sqft=sqft,
        bath=bath,
        bhk=bhk
    )

if __name__ == '__main__':
    app.run(debug=True)
