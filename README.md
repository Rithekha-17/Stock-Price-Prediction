# Stock Price Prediction using Machine Learning

This project predicts **Stock 1** prices based on **Stock 2, Stock 3, Stock 4, and Stock 5** using multiple regression models. The system is trained, evaluated, and deployed as a production-ready web application using **Google Cloud Run**.

---

##  Live App
 [Click here to view the deployed web application](https://stock-predict-937887705926.asia-south1.run.app/)

---

##  Tech Stack

- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib)
- **Machine Learning Models:**
  - Ridge Regression
  - Lasso Regression
  - Polynomial Regression
  - Random Forest Regressor
- **Feature Selection:** `SelectKBest` using `f_regression` 
- **Deployment:** Google Cloud Run (FastAPI / Flask backend)

---

##  Machine Learning Workflow

1. **Data Preprocessing**
   - Missing values handled using mean imputation
   - Duplicate rows removed
   - Outliers removed using IQR method
   - Features scaled using `StandardScaler`

2. **Feature Selection**
   - Used `SelectKBest(f_regression)` to select top 3 predictive features from ['Stock_2', 'Stock_3', 'Stock_4', 'Stock_5']

3. **Model Training**
   - Ridge, Lasso, Polynomial, and Random Forest regressors trained on cleaned data
   - Polynomial model uses degree 2

4. **Evaluation**
   - Metrics: MAE, MSE, RMSE, R²
   - Evaluated on:
     - Training Set
     - 5-Fold Cross-Validation
     - Test Set

5. **Best Model**
   - Based on test performance, **Random Forest** performed best

---

##  Deployment

This project is deployed to **Google Cloud Run**:

- Backend exposes an endpoint to receive input features and return predicted `Stock_1` value
- Accepts JSON or form input
- Hosted publicly using Cloud Run service

 [Live Link](https://stock-predict-937887705926.asia-south1.run.app/)

---



