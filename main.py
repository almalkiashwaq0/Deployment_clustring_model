from fastapi import FastAPI
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel

model = joblib.load('kmeans_model.joblib')
app = FastAPI()
scaler = StandardScaler()
class InputFeatures(BaseModel):
    appearance: int
    goals : float
    award : int
    height : float

def preprocessing(input_features: InputFeatures):
    dict_f = {
            'Appearance': input_features.appearance,
            'Goals' : input_features.goals,
            'award' : input_features.award,
            'height' : input_features.height
            }
    
    return dict_f

@app.post("/predict")
def predict(input_features: InputFeatures):
    data = preprocessing(input_features)  # Get the processed input
    y_pred = model.predict([list(data.values())])  # Make a prediction
    return {"pred": y_pred.tolist()[0]}