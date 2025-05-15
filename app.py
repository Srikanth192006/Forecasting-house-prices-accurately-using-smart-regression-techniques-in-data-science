from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = "model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Train the model first by running train_model.py")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Home page with a form
@app.route('/')
def home():
    html_form = '''
    <h2>House Price Predictor</h2>
    <form method="post" action="/predict">
        Area (sq ft): <input type="number" name="area"><br>
        Bedrooms: <input type="number" name="bedrooms"><br>
        Bathrooms: <input type="number" name="bathrooms"><br>
        Location: <input type="text" name="location"><br>
        Year Built: <input type="number" name="yearBuilt"><br>
        <input type="submit" value="Predict Price">
    </form>
    '''
    return render_template_string(html_form)

# Handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form or request.get_json()
        area = float(data['area'])
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        location = hash(data['location']) % 1000
        yearBuilt = int(data['yearBuilt'])

        features = np.array([[area, bedrooms, bathrooms, location, yearBuilt]])
        prediction = model.predict(features)[0]
        price = round(prediction, 2)

        if request.is_json:
            return jsonify({'predicted_price': price})
        else:
            return f"<h3>Predicted House Price: ${price}</h3><a href='/'>Back</a>"

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT
    app.run(host="0.0.0.0", port=port, debug=True)
