from flask import Flask, request, jsonify

app = Flask(__name__)

# Endpoint for Linear Regression model
@app.route('/linear_regression', methods=['POST'])
def predict_linear_regression():
    data = request.json
    prediction = lr_model.predict([data['features']])[0]
    return jsonify({'prediction': prediction})

# Endpoint for Random Forest model
@app.route('/random_forest', methods=['POST'])
def predict_random_forest():
    data = request.json
    prediction = rf_model.predict([data['features']])[0]
    return jsonify({'prediction': prediction})

# Endpoint for support vector model
@app.route('/support_vector', methods=['POST'])
def predict_support_vector():
    data = request.json
    prediction = rf_model.predict([data['features']])[0]
    return jsonify({'prediction': prediction})

# Ngrok URLs for Linear Regression and Random Forest models
lr_ngrok_url = "http://<ngrok_url>/linear_regression"
rf_ngrok_url = "http://<ngrok_url>/random_forest"
spv_ngrok_url = "http://<ngrok_url>/support_vector"

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)