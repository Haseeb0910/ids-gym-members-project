Calories Burned Prediction 🔥

A small end‑to‑end machine learning project that predicts calories burned during gym workouts using a linear regression model and an interactive Streamlit app. The project covers data cleaning, exploratory data analysis (EDA), model training, and deployment.

Project overview
Dataset: Gym members exercise tracking data (age, gender, height, weight, heart rate, session duration, workout type, etc.). 

Task: Regression – predict Calories_Burned per workout session. 
Model: Scikit‑learn LinearRegression trained on engineered features such as BMI and one‑hot encoded categorical variables.​
Metrics: Evaluated with Mean Absolute Error (MAE) and R² on a hold‑out test set.​
UI: Streamlit app with multiple pages (Introduction, EDA, Prediction, Conclusion). 

Repository structure
data/ – raw and preprocessed CSV files used for analysis and the app.
models/ – saved regression model (calories_model.joblib) for deployment.​
notebooks/
data_loading_cleaning.ipynb – load data, handle missing values, create BMI, and export clean data. 
eda.ipynb – visualizations (distributions, relationships, correlation heatmap). 
preprocessing.ipynb – feature engineering and encoding for model training. 
model.ipynb – train/test split, LinearRegression training, evaluation, and model saving.​
app.py – local Streamlit app entry point. [file:4e222228-0fe7-43c4-a860-04d4f27b0e17]
requirements.txt – Python dependencies (Streamlit, pandas, numpy, scikit‑learn, matplotlib, seaborn, joblib). 

How to run the app locally
Install dependencies:
bash
pip install -r requirements.txt
Run the Streamlit app:

bash
streamlit run app.py
Open the URL shown in the terminal to use the web interface.

The app lets you:
Explore the dataset via EDA plots (distributions, scatterplots, boxplots, correlation heatmap). 
Enter your own workout details to get an estimated calories‑burned value in kcal. 

Model details
Features: Age, BMI, weight, heart‑rate statistics, session duration (hours), workout frequency, gender, workout type, and experience level (with one‑hot encoded columns like Gender_Male and Workout_Type_HIIT). ​
Target: Calories_Burned.​
Training: Scikit‑learn LinearRegression on an 80/20 train‑test split.​
Evaluation: MAE and R² are printed in notebooks/model.ipynb and summarized in the Conclusion page of the app. ​

Live demos
Streamlit Community Cloud: link in repository description.​
Hugging Face Space: https://huggingface.co/spaces/Haseeb0910/calories-burned-prediction.
