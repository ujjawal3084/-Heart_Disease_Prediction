from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained heart disease model
model_path = 'heart_disease_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        int_features = [float(x) for x in request.form.values()]  # Convert to float for medical data
        final_features = [np.array(int_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Has Heart Disease' if prediction[0] == 1 else 'No Heart Disease'

        return render_template('index.html', prediction_text=f'Prediction: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    print("Starting Flask Server... Visit http://127.0.0.1:5000")
    app.run(debug=True)
