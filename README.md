# Credit_Risk_borrower_default_prediction
To compute the probability for a borrower to default in any loan obligation in the upcoming 12 months,

Business Insight Document
Project Overview
The project involves predicting the default status of loans based on customers' financial information. The data set contains information about the customer's credit history, payment details, and demographic information. The objective of this project is to build a predictive model that can accurately classify whether a customer will default on a loan or not.
Data Preprocessing
The data was preprocessed by handling missing values, encoding categorical variables, scaling numerical features, and splitting the data into training and testing sets. The missing values were imputed using the mean or mode of the respective column. The categorical variables were encoded using OneHotEncoder or OrdinalEncoder. The numerical features were scaled using StandardScaler. The data was split into 70% training and 30% testing sets.
1. It defines the list of columns for each of the three types of transformers:
   - `or_columns` for ordinal encoding
   - `oh_columns` for one-hot encoding
   - `transform_columns` for power transformation
2. It imports the necessary transformers from Scikit-learn: `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`, and `PowerTransformer`.
3. It creates a pipeline for the power transformer using `PowerTransformer` from Scikit-learn with the 'yeo-johnson' method.
4. It creates a `ColumnTransformer` that applies each transformer to its respective columns:
   - `OneHotEncoder` for `oh_columns`
   - `OrdinalEncoder` for `or_columns`
   - `transform_pipe` for `transform_columns`
   - `StandardScaler` for numerical columns
5. It fills missing values in the `REPAYMENT_FREQUENCY` column with the mode.
6. It replaces missing values in the `NUMBER_OF_INSTALLMENTS` column with the median.
7. It replaces missing values in the `TOT_OUTSTD_BAL` column with the median.
8. It replaces missing values in the `OVER_DUE_AMT` column with the median.
9. It replaces missing values in the `AMOUNT_OF_INSTALLMENT` column with the median.
10. It replaces missing values in the `INSTALMENT_LOAN_TYPE` column with the mode.
11. It replaces missing values in the `LAST_PAYMENT_AGE` column with the median.
Exploratory Data Analysis
The EDA revealed some interesting insights about the data set. The majority of the customers had not defaulted on their loans. The repayment frequency was predominantly monthly, followed by weekly and quarterly. The majority of the loans had a duration of 1 year or less. The outstanding balance of the loans varied widely, with some customers having significantly high balances.
Most of the loans in the dataset have been successfully repaid, which indicates that the lending business has a good track record of loan repayment.
The majority of the loans in the dataset have been sanctioned for a period of 65 months.
Most of the borrowers have an account age between 2 to 5 years.
The currency used for most of the loans in the dataset is Kenyan Shilling.
There is a strong positive correlation between the amount of installment and the total outstanding balance of the loan.
         Correlation coefficient: 0.40971004398233546
      6.   There is also a strong positive correlation between the age of the loan and the outstanding balance.
      7.   Loans with higher sanction amounts tend to have a longer loan duration.
    8. The repayment frequency of most loans in the dataset is monthly.
    9.    The default rate for loans in the dataset is around 8%, which indicates that there is a risk of default associated with lending.
    10.   Overall, the dataset provides valuable insights into the lending business, which can be used to optimize lending strategies and minimize the risk of default.



Model Development
Three models were trained and tested on the data set:  Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, K-Neighbors Classifier, XGBClassifier, CatBoosting Classifier, Support Vector Classifier, and AdaBoost Classifier.
The Random Forest and Decision Tree models seem to have overfitted the training data since they achieve perfect accuracy, F1 score, precision, recall, and ROC AUC scores on the training set. However, their performance on the test set is still good, with accuracy, F1 score, precision, recall, and ROC AUC scores above 0.99.
Gradient Boosting and K-Neighbors Classifier have slightly lower performance metrics on the test set compared to the Random Forest and Decision Tree models, but they have better generalization since their performance metrics on the training set are lower than 1.0.
Logistic Regression, CatBoosting Classifier, Support Vector Classifier, and AdaBoost Classifier have the lowest performance metrics on the test set, but their generalization is better than the Random Forest and Decision Tree models since they have lower performance metrics on the training set.
It might be worth considering using the K-Neighbors Classifier, XGBClassifier, or Gradient Boosting for this particular classification task since they have a good balance between high-performance metrics on the test set and generalization to unseen data.
Done some hyperparameter tuning parameters for XGBoost and Random Forest models. 
Conclusion Based on the results of the three models - Random Forest Classifier, KNeighborsClassifier, and XGBClassifier - we can conclude that all three models performed well on both the training and test sets, with high accuracy, F1 score, precision, and recall.
However, the Random Forest Classifier achieved the highest scores on the test set, with an accuracy of 0.9969, an F1 score of 0.9969, a precision of 0.9938, and a recall of 1.0000. The XGBClassifier also performed well with an accuracy of 0.9978, an F1 score of 0.9978, a precision of 0.9956, and a recall of 1.0000.
Therefore, we can conclude that the Random Forest Classifier is the best model for this task, as it achieved the highest scores on the test set.


