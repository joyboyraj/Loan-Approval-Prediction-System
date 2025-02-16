from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('random_forest_best_model.pkl')  # Replace with your actual model path
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    try:
        name = request.form['name']
        age = int(request.form['age'])
        employment_type = request.form['employment_type']
        loan_purpose = request.form['loan_purpose']
        cibil_score = int(request.form['cibil_score'])
        loan_term = int(request.form['loan_term'])
        loan_amount = float(request.form['loan_amount'])
        income_annum = float(request.form['income_annum'])
        luxury_assets_value = float(request.form['luxury_assets_value'])

        input_data = pd.DataFrame({
            'cibil_score': [cibil_score],
            'loan_term': [loan_term],
            'loan_amount': [loan_amount],
            'income_annum': [income_annum],
            'luxury_assets_value': [luxury_assets_value]
        })

        prediction = model.predict(input_data)[0]
        loan_status = 'Approved' if prediction == 1 else 'Rejected'

        return redirect(url_for('results', name=name, age=age, employment_type=employment_type,
                               loan_purpose=loan_purpose, loan_status=loan_status))

    except (ValueError, KeyError) as e:
        return render_template('error.html', error_message="Invalid input. Please check your form.")
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('error.html', error_message="An error occurred during prediction. Please try again.")

@app.route('/results')
def results():
    name = request.args.get('name')
    age = request.args.get('age')
    employment_type = request.args.get('employment_type')
    loan_purpose = request.args.get('loan_purpose')
    loan_status = request.args.get('loan_status')

    return render_template('result.html', name=name, age=age, employment_type=employment_type,
                           loan_purpose=loan_purpose, loan_status=loan_status)

if __name__ == '__main__':
    app.run(debug=True)