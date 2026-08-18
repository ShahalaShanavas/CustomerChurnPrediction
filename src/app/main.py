"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: web framework for Python designed mainly for building APIs quickly and efficiently. FastAPI automatically generates interactive documentation.
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Pydantic is a Python library used to validate, parse, and structure data. It's especially important in FastAPI, because FastAPI uses Pydantic to make sure the data coming into your API has the correct format.

"""


from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict

##### App initialization #####
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)  
#Creates the FastAPI app instance and gives it some information that will appear in the API documentation.
#The title/description/version args don't affect behavior — they populate the auto-generated Swagger docs at /docs.

##### Health check endpoint ######
@app.get("/")
def root():
    return {"status": "ok"}

###### Request schema (CustomerData) ######
class CustomerData(BaseModel):

    """
    CustomerData is a Python class representing one customer's information.
    BaseModel comes from Pydantic.By inheriting from BaseModel, Pydantic automatically validates the data you send to this model.
    This pydantic model defines the exact shape of a valid request body for the churn prediction API. 
    The fields in this model correspond to the features used by the machine learning model to make predictions. 
    It ensures that the incoming data is validated and structured correctly before being processed by the prediction function.

    """

    # Demographics
    gender: str                # "Male" or "Female"
    Partner: str               # "Yes" or "No" - has partner
    Dependents: str            # "Yes" or "No" - has dependents
    
    # Phone services
    PhoneService: str          # "Yes" or "No"
    MultipleLines: str         # "Yes", "No", or "No phone service"
    
    # Internet services  
    InternetService: str       # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str        # "Yes", "No", or "No internet service"
    OnlineBackup: str          # "Yes", "No", or "No internet service"
    DeviceProtection: str      # "Yes", "No", or "No internet service"
    TechSupport: str           # "Yes", "No", or "No internet service"
    StreamingTV: str           # "Yes", "No", or "No internet service"
    StreamingMovies: str       # "Yes", "No", or "No internet service"
    
    # Account information
    Contract: str              # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str      # "Yes" or "No"
    PaymentMethod: str         # "Electronic check", "Mailed check", etc.
    
    # Numeric features
    tenure: int                # Number of months with company
    MonthlyCharges: float      # Monthly charges in dollars
    TotalCharges: float        # Total charges to date


    #Each field's type annotation (str, int, float) is enforced automatically by Pydantic — 
    #if a client sends tenure: "abc", FastAPI returns a 422 validation error before the code even runs.

###### Prediction endpoint ######

@app.post("/predict")
def get_prediction(data: CustomerData): 
    try:
        result = predict(data.model_dump())
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
    
#When someone sends a POST request to /predict, the function below is executed.
#data: CustomerData tells FastAPI that the incoming request should follow the CustomerData schema 
#model_dump() converts the Pydantic model into a normal Python dictionary since the predict function expects a dictionary as input. 
#predict(...) performs preprocessing, and uses the trained machine-learning model defined in src/serving/inference.py to generate a prediction.

# =================================================== # 

###### Gradio Web Interface ######

"""
Gradio interface function that processes form inputs and returns prediction.

This function:
1. Takes individual form inputs from Gradio UI. 
2. Constructs the data dictionary matching the API CustomerData schema
3. Calls the same inference pipeline used by the API
4. Returns user-friendly prediction string

"""

def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    # Construct the data dictionary matching the API CustomerData schema
    # value of each input received from the Gradio interface is assigned to the corresponding key in the dictionary.
    
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),              # Ensure integer type
        "MonthlyCharges": float(MonthlyCharges),  # Ensure float type
        "TotalCharges": float(TotalCharges)       # Ensure float type
    }

    # Call the same inference pipeline used by the API
    result = predict(data)
    return f"Prediction: {result}"  # Return user-friendly prediction string


####### Gradio UI Configuration ######

#builds a Gradio web interface for a Telco Customer Churn prediction model 
#and then connects that interface to a FastAPI application.


demo = gr.Interface(
    fn=gradio_interface,  # The function to call when the form is submitted
    inputs=[
        # Demographics section
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),

        # Phone services section
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),

        # Internet services section
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="DSL"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="No"),

        # Contract and billing section
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", 
                     "Credit card (automatic)"], label="Payment Method", value="Electronic check"),    

        # Numeric features section
        gr.Number(label="Tenure (months)", value=1,minimum=0, maximum=100),
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0, maximum=200),
        gr.Number(label="Total Charges ($)", value=85.0, minimum=0, maximum=10000)
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="🔮 Telco Customer Churn Predictor",
    description="""
    **Predict customer churn probability using machine learning**

    Fill in the customer details below to get a churn prediction. The model uses LightGBM trained on historical telecom customer data to identify customers at risk of churning.
    
    """,
    examples=[
        # This creates example inputs that users can click to automatically populate the form.
        # This is useful because the user doesn't have to manually enter all 17 fields.

        # High churn risk example
        ["Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No", 
         "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 
         1, 85.0, 85.0],
        # Low churn risk example  
        ["Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
         "Yes", "No", "No", "Two year", "No", "Credit card (automatic)",
         60, 45.0, 2700.0]
        ])
demo.theme = gr.themes.Soft() # controls the visual appearance of the interface. provides a softer, more polished default Gradio appearance.

#### FastAPI integration  === MOUNT GRADIO UI INTO FASTAPI === ####
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI

app = gr.mount_gradio_app(
    app,           # FastAPI application instance
    demo,          # Gradio interface
    path="/ui"     # URL path where Gradio will be accessible
)


if __name__ == "__main__":
      import uvicorn #uvicorn is the server that runs FastAPI application.
      uvicorn.run(app, host="0.0.0.0", port=8000) 
      #This starts Uvicorn. Tells Uvicorn to Run the FastAPI application stored in the variable app that contains the FastAPI application with the Gradio UI mounted onto it.
    

#### run the script using the command : python -m src.app.main

#http://localhost:8000/	Health check → {"status": "ok"}
#http://localhost:8000/docs	Swagger UI 
#http://localhost:8000/ui	Gradio form

