from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ValidationError

app = FastAPI()

# Load the model
def load_model(filename):
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model file '{filename}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CustomerData(BaseModel):
    CreditScore: int = Field(ge=0, le=850)
    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=0)
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float = Field(ge=0)
    Geography: str = Field(pattern='^(France|Germany|Spain)$')
    Gender: str = Field(pattern='^(Male|Female)$')

class PredictionRequest(BaseModel):
    model: str
    data: CustomerData

def process_data(customer_data):
    data = pd.DataFrame(customer_data.dict(), index=[0]).iloc[0]

    input_dict = {
        "CreditScore": data["CreditScore"],
        "Age": data["Age"],
        "Tenure": data["Tenure"],
        "Balance": data["Balance"],
        "NumOfProducts": data["NumOfProducts"],
        "HasCrCard": data["HasCrCard"],
        "IsActiveMember": data["IsActiveMember"],
        "EstimatedSalary": data["EstimatedSalary"],
        "Geography_France": 1 if data["Geography"] == "France" else 0,
        "Geography_Germany": 1 if data["Geography"] == "Germany" else 0,
        "Geography_Spain": 1 if data["Geography"] == "Spain" else 0,
        "Male": 1 if data["Gender"] == "Male" else 0,
        "CLV": data['Balance'] * data['EstimatedSalary'] / 10000,
        "TenureAgeRatio": data['Tenure'] / data['Age'],
        "AgeGroup_MiddleAge": 1 if data['Age'] >= 30 and data['Age'] < 45 else 0,
        "AgeGroup_Senior": 1 if data['Age'] >= 45 and data['Age'] < 60 else 0,
        "AgeGroup_Elderly": 1 if data['Age'] >= 60 else 0
    }

    input_df = pd.DataFrame(input_dict, index=[0])
    return input_df

def process_data_selective(customer_data):
    data = pd.DataFrame(customer_data.dict(), index=[0]).iloc[0]

    input_dict = {
        "Age": data["Age"],
        "NumOfProducts": data["NumOfProducts"],
        "IsActiveMember": data["IsActiveMember"],
        "Geography_Germany": 1 if data["Geography"] == "Germany" else 0,
        "AgeGroup_Senior": 1 if data['Age'] >= 45 and data['Age'] < 60 else 0
    }

    input_df = pd.DataFrame(input_dict, index=[0])
    return input_df
    
def get_prediction(request):
    model_name = request.model
    if model_name == 'xgb':
        model = load_model('xgb_smote.pkl')
    elif model_name == 'rf':
        model = load_model('rf_smote.pkl')
    elif model_name == 'gb-selective':
        model = load_model('gradientBoosting_selective.pkl')
    elif model_name == 'voting':
        model = load_model('voting_clf.pkl')
    elif model_name == 'stacking':
        model = load_model('stacking_smote.pkl')
    else:
        raise HTTPException(status_code=400, detail=f"Invalid model name: '{model_name}'. Please choose one of the following: 'xgb', 'rf', 'gb-selective', 'voting', 'stacking'.")

    if model_name == 'gb-selective':
        input_df = process_data_selective(request.data)
    else:
        input_df = process_data(request.data)
    
    try:
        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return prediction, probabilities

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        prediction, probabilities = get_prediction(request)
        return {
            "prediction": prediction.tolist()[0],
            "probabilities": probabilities.tolist()
        }
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
