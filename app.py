from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import os

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load label encoders
le1, le2, le3, le4, le5, le6, le7, le8, le9 = (
    pickle.load(open(f'le{i}.pkl', 'rb')) for i in range(1, 10)
)

# Load the unique airlines list
with open("airline_unique.json", 'r') as f:
    airline_unique = json.load(f)

# Load the unique origins list
with open("origin.json", 'r') as f:
    origin_unique = json.load(f)

# Load the unique destinations list
with open("destination.json", 'r') as f:
    destination_unique = json.load(f)

# Load the unique months list
with open("months.json", 'r') as f:
    months_unique = json.load(f)
    
# Load StandardScaler
ss = pickle.load(open('ss.pkl', 'rb'))  

@app.route('/')
def home():
    # return render_template('index.html', airlines=airline_unique)
    return render_template('index.html', airlines=airline_unique, origins=origin_unique,destinations=destination_unique,months=months_unique)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        input_data = {
            'Airline Name': request.form['airline'],
            'Overall_Rating': float(request.form['overall_rating']),
            'Verified': request.form['verified'],
            'Type Of Traveller': request.form['traveller_type'],
            'Seat Type': request.form['seat_type'],
            'Origin': request.form['origin'],
            'Destination': request.form['destination'],
            'Month Flown': request.form['month_flown'],
            'Year Flown': request.form['year_flown'],
            'Seat Comfort': float(request.form['seat_comfort']),
            'Cabin Staff Service': float(request.form['cabin_service']),
            'Food & Beverages': float(request.form['food_beverages']),
            'Ground Service': float(request.form['ground_service'])
        }

        # Fit and transform label encoders
        for feature, encoder in zip(input_data.keys(), [le1, le2, le3, le4, le5, le6, le7, le8, le9]):
            input_data[feature] = encoder.fit_transform([input_data[feature]])[0]

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Standardize input
        input_df = ss.transform(input_df)

        # Make prediction
        prediction = model.predict(input_df)

        # return render_template('index.html', prediction_text=f'The prediction is {prediction[0]}')
        prediction_text = 'Recommended' if prediction[0] == 1 else 'Not Recommended'

        return render_template('index.html', prediction_text=f'{prediction_text}')


if __name__ == '__main__':
    app.run(debug=True)
