from pandas.core import base
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 15px 32px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 10px;
        border: none;
        transition: background-color 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key=os.environ.get('GROQ_API_KEY')
)


def explain_prediction(probability, input_dict, surname):
  prompt = f"""You are an expert data scientist at a bank, where you specialize in 
  interperting and explaining predictions of machine learning models.

  Your machine learning model has predicted that a customer named {surname} has a  
  {round(probability * 100, 1)}% probablity of churning, based on the information 
  provided below.

  Here is the customer's information:
  {input_dict}

  Here are the machine learning model's top 10 most important features for predicting 
  churn:

            Feature | Importance
  ------------------------------------------
      NumOfProducts | 0.323888
     IsActiveMember | 0.164146
                Age | 0.109550
  Geography_Germany | 0.091373
            Balance | 0.052786
   Geography_France |	0.046463
               Male | 0.045283
    Geography_Spain | 0.036855
        CreditScore | 0.035005
    EstimatedSalary	| 0.032655
          HasCrCard	| 0.031940
     Tenure (years) | 0.030054

  {pd.set_option('display.max_columns', None)}

  Here are summary statistics for churned customers:
  {df[df['Exited'] == 1].describe()}

  Here are summarry statistics for non-churned customers:
  {df[df['Exited'] == 0].describe()}

  - If the customer has over a 40% risk of churning, generate a 3 sentence 
  explaination of why they are at risk of churning.
  
  - If the customer has less than a 40% risk of churning, generate a 3 sentence 
  explaination of why they might not be at risk of churning. 
  
  - Your explaination should be based on the customer's information, the summary 
  statistics of churned and non-churned customers, and the featuer importances 
  provided.

  DO NOT mention the risk threshold at all. The reader does not care about that.

  Don't mention the probability of churning, or the machine learning model, or say 
  anything like "Based on the machine learning model's prediction and top 10 most 
  important features", just explain the prediction.

  Also Do NOT use the feature names in your explanation, make sure to translate them to 
  layman terms. For example: instead of saying NumOfProducts, just say number of 
  products. The reader is not a data scientist, do not try to use the actual feature 
  names.

  DO NOT say something like: "{surname} has a {probability}% probability of churning. Here is an explanation:" Just go straight into the explanation. The reader DOES NOT have time to waste.

  When you are explaining, use actual values from the customer's information, not just general statements. For example: instead of saying "they hold a relatively low number of products", say "they only have 2 products."


  """

  print("EXPLAINATION PROMPT", prompt)

  raw_response = client.chat.completions.create(
    model="llama-3.2-3b-preview",
    messages=[{
      "role": "user",
      "content": prompt
    }]
  )

  return raw_response.choices[0].message.content



def generate_email(probability, input_dict, explanation, surname):
  prompt = f"""You are a manager at a bank responsible for ensuring customer retention with various offers.

  You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

  Here is the customer's information:
  {input_dict}

  Here is some explanation as to why the customer might be at risk of churning:
  {explanation}

  Generate an email to the customer asking them to stay, or offering them incentives to become more loyal to the bank.

  List the incentives in an appropariate markdown bullet point format.

  **Instructions for Bullet Points:**
  - List out 2 to 5 specific and detailed offers in bullet point format. 
  - Do not mention the probability of churning or the machine learning model.

  Examples of bullet points:
  - We will increase your credit score limit to $2000.
  - You will receive a personalized financial consultation.

  Keep it personal but professional. Don't forget to sign off at the end.
  DO NOT mention the word 'incentive' in the email.
  """


  raw_response = client.chat.completions.create(
    model='llama-3.2-3b-preview',
    messages=[{
      "role": "user",
      "content": prompt
    }]
  )

  print("\n\nEMAIL PROMPT", prompt)

  return raw_response.choices[0].message.content
  


def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)

xgb_model = load_model('xgb_smote.pkl')
gradientBoosting_model = load_model('gradientBoosting_selective.pkl')
rf_model = load_model('rf_smote.pkl')
voting_model = load_model('voting_clf.pkl')
stacking_model = load_model('stacking_smote.pkl')

def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, 
                  has_credit_card, is_active_member, estimated_salary):
  
    input_dict = {
      'CreditScore': credit_score,
      'Age': age,
      'Tenure': tenure,
      'Balance': balance,
      'NumOfProducts': num_products,
      'HasCrCard': int(has_credit_card),
      'IsActiveMember': int(is_active_member),
      'EstimatedSalary': estimated_salary,
      'Geography_France': 1 if location == 'France' else 0,
      'Geography_Germany': 1 if location == 'Germany' else 0,
      'Geography_Spain': 1 if location == 'Spain' else 0,
      "Male": 1 if gender == 'Male' else 0,
      "CLV": balance * estimated_salary / 100000,
      "TenureAgeRatio": tenure / age,
      "AgeGroup_MiddleAge": 1 if age >= 30 and age < 45 else 0,
      "AgeGroup_Senior": 1 if age >= 45 and age < 60 else 0,
      "AgeGroup_Elderly": 1 if age >= 60 else 0
    }
  
    input_df = pd.DataFrame([input_dict])
  
    # for models with feature selection
    selected_columns = ['Age', 'NumOfProducts', 'IsActiveMember', 'Geography_Germany', 'AgeGroup_Senior']
    selective_input_dict = {key: input_dict[key] for key in selected_columns}
    selective_input_df = pd.DataFrame([selective_input_dict])

    return input_df, input_dict, selective_input_df, selective_input_dict



def make_predictions(input_df, selective_input_df, surname):
  probabilities = {
    'Gradient Boosting': gradientBoosting_model.predict_proba(selective_input_df)[0][1],
    'Voting Classifier': voting_model.predict_proba(input_df)[0][1],
    'Stacking Classifier': stacking_model.predict_proba(input_df)[0][1],
    'Random Forest': rf_model.predict_proba(input_df)[0][1],
    'XGBoost': xgb_model.predict_proba(input_df)[0][1]

  }

  avg_probability = np.mean(list(probabilities.values()))

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig,use_container_width=True)
    st.write(f"{surname} has a {avg_probability:.2%} probability of churning.")
    clv = input_df['CLV'].iloc[0]
    if clv > 0:
      st.write(f"Estimated Customer Lifetime Value: {clv:.2f}")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs,use_container_width=True)

  return avg_probability

def generate_percentiles(df, input_dict):
  all_num_products = df['NumOfProducts'].sort_values().tolist()
  all_balances = df['Balance'].sort_values().tolist()
  all_estimated_salaries = df['EstimatedSalary'].sort_values().tolist()
  all_tenures = df['Tenure'].sort_values().tolist()
  all_credit_scores = df['CreditScore'].sort_values().tolist()

  product_rank = np.searchsorted(all_num_products, input_dict['NumOfProducts'], side='right')
  balance_rank = np.searchsorted(all_balances, input_dict['Balance'], side='right')
  salary_rank = np.searchsorted(all_estimated_salaries, input_dict['EstimatedSalary'], side='right')
  tenure_rank = np.searchsorted(all_tenures, input_dict['Tenure'], side='right')
  credit_rank = np.searchsorted(all_credit_scores, input_dict['CreditScore'], side='right')

  
  N = 10000

  percentiles = {
    'CreditScore': int(np.floor((credit_rank / N) * 100)),
    'Tenure': int(np.floor((tenure_rank / N) * 100)),
    'EstimatedSalary': int(np.floor((salary_rank / N) * 100)),
    'Balance': int(np.floor((balance_rank / N) * 100)),
    'NumOfProducts': int(np.floor((product_rank / N) * 100)),
  }


  fig = ut.create_percentile_chart(percentiles)
  st.plotly_chart(fig,use_container_width=True)
  

  return percentiles
  
  # percentile = (n - 0.5 / N) * 100



st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
  
  selected_customer_id = int(selected_customer_option.split(" - ")[0])
  print("Selected Customer ID", selected_customer_id)

  selected_surname = selected_customer_option.split(" - ")[1]
  print("Surname", selected_surname)
  
  selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]
  print("Selected Customer", selected_customer)

  col1, col2 = st.columns(2)

  with col1:

    credit_score = st.number_input(
      "Credit Score",
      min_value=300,
      max_value=850,
      value=int(selected_customer['CreditScore'])
    )

    print('geo', selected_customer['Geography'])

    location = st.selectbox(
      "Location", ["Spain", "France", "Germany"],
      index=["Spain", "France", "Germany"].index(
        selected_customer['Geography']))

    gender = st.radio("Gender", ["Male", "Female"], index=0 if 
                      selected_customer['Gender'] == "Male" else 1)

    age = st.number_input("Age",
            min_value=18,
            max_value=100,
            value=int(selected_customer['Age']))

    tenure = st.number_input("Tenure (years)",
              min_value=0,
              max_value=50,
              value=int(selected_customer['Tenure']))

  with col2:

    balance = st.number_input(
      "Balance",
      min_value=0.0,
      value=float(selected_customer['Balance']))

    num_products = st.number_input(
      "Number of Products",
      min_value=0,
      max_value=10,
      value=int(selected_customer['NumOfProducts']))

    has_credit_card = st.checkbox(
      "Has Credit Card",
      value=bool(selected_customer['HasCrCard']))

    is_active_member = st.checkbox(
      "Is Active Member",
      value=bool(selected_customer['IsActiveMember']))

    estimated_salary = st.number_input(
      "Estimated Salary",
      min_value=0.0,
      value=float(selected_customer['EstimatedSalary']))

  input_df, input_dict, selective_input_df, selective_input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

  percentiles = generate_percentiles(df, input_dict)

  if st.button('Are they churning?', use_container_width=True):
    with st.spinner("Predicting..."):
      avg_probability = make_predictions(input_df, selective_input_df, selected_customer['Surname'])

    with st.spinner("Explaining..."):
      explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])
  
  
    st.markdown("---")
  
    st.subheader("Explanation of Prediction")
  
    st.markdown(explanation)

    with st.spinner("Writing an email..."):
      email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])
  
    st.markdown("---")
  
    st.subheader("Personalized Email")
  
    st.markdown(email)

    st.markdown("---") 
    
    st.subheader("API")

    st.markdown("""
        <div style='display: flex; justify-content: center; align-items: center; margin-top: 20px;'>
            <a href="https://github.com/rafiks7/Churn-ML-Models" target="_blank" 
            style='background-color: #4CAF50; padding: 10px 20px; color: white; text-align: center; 
            text-decoration: none; border-radius: 8px; font-size: 18px;'>
                Use models in your app
            </a>
        </div>
        """, unsafe_allow_html=True)