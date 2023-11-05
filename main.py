from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import joblib
import pandas as pd

# Creating our app
app = FastAPI()

# Define CORS middleware
origins = ["*"]
methods = ["*"]
headers = ["*"]

#load my groundHog_model
groundHog_model = joblib.load("groundHog_model.pkl") 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)


# Create a prediction endpoint
@app.get("/predict/")
def predict_shadow(
    year: int, 
    feb_temp: float, 
    mar_temp: float, 
    northeast_feb: float = 0.0,  # Provide default values
    midwest_feb: float = 0.0,
    penns_feb: float = 0.0,
    northeast_mar: float = 0.0,
    midwest_mar: float = 0.0,
    penns_mar: float = 0.0
):
    
    # Prepare new data for prediction
    new_data = pd.DataFrame({
        'Year': [year],
        'February Average Temperature': [feb_temp],
        'February Average Temperature (Northeast)': [northeast_feb],
        'February Average Temperature (Midwest)': [midwest_feb],
        'February Average Temperature (Pennsylvania)': [penns_feb],
        'March Average Temperature': [mar_temp],
        'March Average Temperature (Northeast)': [northeast_mar],
        'March Average Temperature (Midwest)': [midwest_mar],
        'March Average Temperature (Pennsylvania)': [penns_mar]
    })

    # Make predictions
    predicted_class = groundHog_model.predict(new_data)

    return {"Predicted Shadow": predicted_class[0]}

# Implement default API endpoint
@app.get("/")
def read_root():
    return {"message": "FastAPI setup successful"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
