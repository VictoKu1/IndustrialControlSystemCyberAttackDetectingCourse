from Industrial_power_control_system_cyber_attacks_detetection_ui import calculate
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pandas as pd
import joblib


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Read the CSV data into a DataFrame
        df = pd.read_csv(file)

        results01 = calculate(df)

        # Store the predictions in a session variable
        session['predictions'] = results01

        # Redirect to the results page
        return redirect(url_for('results'))

    return render_template('index.html')

@app.route("/results")
def results():
    # Get the predictions from the session variable
    predictions = session['predictions']

    # Clear the session variable to prevent data leakage
    session.pop('predictions', None)

    # Return the predictions as a list of strings
    return render_template('results.html', predictions=predictions)

if __name__ == "__main__":
    # Run the app
    app.secret_key = 'my-secret-key' # Set a secret key for the session
    app.run(host="0.0.0.0", port=5079)


