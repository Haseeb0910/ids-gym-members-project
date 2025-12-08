import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Basic page config
# ---------------------------
st.set_page_config(
    page_title="Calories Burned Prediction App",
    layout="wide",
)

# ---------------------------
# Load data and model
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/gym_members_exercise_tracking.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/calories_model.joblib")

df = load_data()
model = load_model()

target = "Calories_Burned"

feature_cols = [
    "Age",
    "BMI",
    "Weight (kg)",
    "Max_BPM",
    "Avg_BPM",
    "Session_Duration (hours)",
    "Workout_Frequency (days/week)",
    "Gender_Male",
    "Workout_Type_HIIT",
    "Experience_Level_2",
]

# ---------------------------
# Sidebar navigation
# ---------------------------
st.title("Calories Burned Prediction App")

col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

with col_nav1:
    intro_btn = st.button("Introduction")
with col_nav2:
    eda_btn = st.button("EDA")
with col_nav3:
    pred_btn = st.button("Prediction")
with col_nav4:
    concl_btn = st.button("Conclusion")

# Decide which page to show
if "page" not in st.session_state:
    st.session_state.page = "Introduction"

if intro_btn:
    st.session_state.page = "Introduction"
elif eda_btn:
    st.session_state.page = "EDA"
elif pred_btn:
    st.session_state.page = "Prediction"
elif concl_btn:
    st.session_state.page = "Conclusion"

page = st.session_state.page

# ---------------------------
# 1. Introduction page
# ---------------------------
if page == "Introduction":
    st.title("Calories Burned Prediction App 🔥")
    st.markdown(
        """
        Welcome to the **Calories Burned Prediction App** 💪  
        This app uses a linear regression model to estimate calories burned
        based on your body measurements and workout details.
        """
    )
    st.subheader("Dataset overview")
    st.write("First 5 rows of the raw dataset:")
    st.dataframe(df.head())

    st.markdown(
        """
        **Target variable:** `Calories_Burned`  
        **Example input features:** Age, BMI, Weight, heart rate, session duration, workout frequency, gender, workout type, and experience level.
        """
    )

# ---------------------------
# 2. EDA page 
# ---------------------------
elif page == "EDA":
    st.title("Exploratory Data Analysis 📊")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Calories vs. Session Duration ⏱️")
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x="Session_Duration (hours)",
            y="Calories_Burned",
            hue="Workout_Type",
            alpha=0.7,
            ax=ax,
        )
        st.pyplot(fig)

    with col2:
        st.subheader("Distribution of Calories Burned 🔥")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["Calories_Burned"], bins=30, kde=True, ax=ax2)
        st.pyplot(fig2)

    st.subheader("Calories by Workout Type 🏋️")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x="Workout_Type", y="Calories_Burned", ax=ax3)
    st.pyplot(fig3)

    col3, col4 = st.columns(2)

    with col3: 
        st.subheader("BMI Distribution ⚖️")
        fig4, ax4 = plt.subplots()
        sns.histplot(df["BMI"], bins=30, kde=True, ax=ax4)
        st.pyplot(fig4)

    with col4:
        st.subheader("Avg BPM vs Calories 💪")
        fig5, ax5 = plt.subplots()
        sns.scatterplot(
            data=df,
            x="Avg_BPM",
            y="Calories_Burned",
            hue="Workout_Type",
            alpha=0.7,
            ax=ax5,
    )
        st.pyplot(fig5)
        
    st.subheader("Correlation Heatmap 🔥")

    numeric_cols = [
        "Age",
        "BMI",
        "Weight (kg)",
        "Height (m)",
        "Max_BPM",
        "Avg_BPM",
        "Resting_BPM",
        "Session_Duration (hours)",
        "Fat_Percentage",
        "Water_Intake (liters)",
        "Workout_Frequency (days/week)",
        "Calories_Burned",
    ]

    corr = df[numeric_cols].corr()

    fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax_hm,
    )
    st.pyplot(fig_hm)

# ---------------------------
# 3. Prediction page
# ---------------------------
elif page == "Prediction":
    st.title("Predict Calories Burned 🧮")
    st.markdown("Fill in the workout and body details, then click **Predict** to see the estimated calories burned. 🚴")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=80, value=25)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.10)
        max_bpm = st.number_input("Max BPM", min_value=80, max_value=220, value=170)
        avg_bpm = st.number_input("Avg BPM", min_value=60, max_value=200, value=140)

    with col2:
        session_minutes = st.number_input(
            "Session Duration (minutes)", min_value=10, max_value=300, value=60, step=5
        )
        session_hours = session_minutes / 60.0
        workout_freq = st.number_input(
            "Workout Frequency (days/week)", min_value=1, max_value=7, value=3
        )

        gender = st.radio("Gender", ["Female", "Male"])
        workout_type = st.selectbox(
            "Workout Type", ["Cardio", "HIIT", "Strength", "Yoga"]
        )
        experience = st.selectbox("Experience Level", [1, 2, 3])

    # Map to dummies used in training
    gender_male = 1 if gender == "Male" else 0
    workout_type_hiit = 1 if workout_type == "HIIT" else 0
    experience_level_2 = 1 if experience == 2 else 0

    if st.button("Predict Calories Burned"):
        input_dict = {
            "Age": age,
            "BMI": bmi,
            "Weight (kg)": weight,
            "Max_BPM": max_bpm,
            "Avg_BPM": avg_bpm,
            "Session_Duration (hours)": session_hours,
            "Workout_Frequency (days/week)": workout_freq,
            "Gender_Male": gender_male,
            "Workout_Type_HIIT": workout_type_hiit,
            "Experience_Level_2": experience_level_2,
        }

        input_df = pd.DataFrame([input_dict])

        # Ensure column order matches training
        input_df = input_df[feature_cols]

        pred = model.predict(input_df)[0]   

        st.success(f"Predicted Calories Burned: {pred:.2f} kcal")

# ---------------------------
# 4. Conclusion page
# ---------------------------
elif page == "Conclusion":
    st.title("Conclusion ✅")


    st.markdown(
        """
        - A linear regression model was trained to predict Calories_Burned 
          using gym members' body measurements and workout details.  
        - On the test set, the model achieved a low MAE and a high R², 
          indicating that predictions are close to the true values and 
          most of the variance in calories is explained.  
        - This app can be used as a simple tool to estimate calories burned 
          for different workout scenarios.
        """
    )
