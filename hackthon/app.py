from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the logistic regression model
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        satisfaction_rating = float(request.form['satisfaction_rating'])
        salary = float(request.form['salary'])
        tenure = float(request.form['tenure'])
        
        # Make prediction
        prediction = model.predict([[satisfaction_rating, salary, tenure]])
        
        # Render result.html and pass prediction result as an argument
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
