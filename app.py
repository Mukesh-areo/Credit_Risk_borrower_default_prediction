import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the preprocessor pipeli
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load the best random forest model
with open('best_model_RF.pkl', 'rb') as f:
    forest_model = pickle.load(f)


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

# @app.route('/')
# def home():
#     return 'Loan Status Prediction App'


@app.route('/predict', methods=['POST'])
def predict_loan_status():
    # Get the input data from the request
    input_data = pd.DataFrame({
        'currency_code': [request.form['currency_code']],
        'REPAYMENT_FREQUENCY': [request.form['repayment_frequency']],
        'NUMBER_OF_INSTALLMENTS': [float(request.form['number_of_installments'])],
        'SANCTION_AMT': [float(request.form['sanction_amt'])],
        'TOT_OUTSTD_BAL': [float(request.form['tot_outstd_bal'])],
        'OVER_DUE_AMT': [float(request.form['over_due_amt'])],
        'AMOUNT_OF_INSTALLMENT': [float(request.form['amount_of_installment'])],
        'INSTALMENT_LOAN_TYPE': [request.form['installment_loan_type']],
        'loan_status': [request.form['loan_status']],
        'YEAR_REPORTED': [int(request.form['year_reported'])],
        'MONTH_REPORTED': [int(request.form['month_reported'])],
        'DAY_REPORTED': [int(request.form['day_reported'])],
        'DAY_OF_WEEK_REPORTED': [int(request.form['day_of_week_reported'])],
        'ACCOUNT_AGE': [float(request.form['account_age'])],
        'LAST_PAYMENT_AGE': [float(request.form['last_payment_age'])],
        'LOAN_DURATION': [float(request.form['loan_duration'])]
    })

    # Preprocess the input data using the preprocessor pipeline
    preprocessed_data = preprocessor.transform(input_data)

    # Make the prediction using the random forest model
    prediction = forest_model.predict(preprocessed_data)


    if str(prediction[0])=='1':
        return render_template('index.html',prediction_texts="We are sorry to inform you that based on our analysis, we predict that you are at high risk of defaulting on a loan. Therefore, we cannot approve your loan application at this time.")
    else:
        return render_template('index.html',prediction_text="Congratulations! Based on our analysis, we predict that you are eligible for a loan with low risk of default. Please submit your application to proceed")
    # else:
    #     return render_template('index.html')

    # # Return the prediction to the user
    # return str(prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
