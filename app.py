import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
import warnings
import gradio as gr

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('indian_liver_patient.csv')

# Check for missing values
df.isnull().sum()

# Drop missing values
df1 = df.dropna()
df1.isnull().any()

# Drop correlated features
df2 = df1.drop(columns=['Direct_Bilirubin', 'Alamine_Aminotransferase', 'Total_Protiens'])

# Map target values to binary
df2['Dataset'] = df2['Dataset'].map({1: 0, 2: 1})

# Define the features and target variables
X = df2[['Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Aspartate_Aminotransferase', 'Albumin', 'Albumin_and_Globulin_Ratio']]
y = df2['Dataset']

# Encode the 'Gender' column to numeric
labelencoder = LabelEncoder()
X['Gender'] = labelencoder.fit_transform(X['Gender'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# Create the AdaBoostClassifier model
ADB = AdaBoostClassifier(n_estimators=125, learning_rate=0.6, random_state=42)
ADB.fit(X_train, y_train)

# Function to predict liver disease using the trained model
def predict_liver_disease(Gender, Total_Bilirubin, Alkaline_Phosphotase, Aspartate_Aminotransferase, Albumin, Albumin_and_Globulin_Ratio):
    Gender = labelencoder.transform([Gender])[0]
    input_data = np.array([[Gender, Total_Bilirubin, Alkaline_Phosphotase, Aspartate_Aminotransferase, Albumin, Albumin_and_Globulin_Ratio]])
    scaled_input_data = scaler.transform(input_data)
    prediction = ADB.predict(scaled_input_data)[0]
    return "Positive" if prediction == 1 else "Negative"

# Define the input fields for the Gradio interface
inputs = [
    gr.inputs.Dropdown(["Male", "Female"], label="Gender"),
    gr.inputs.Number(label="Total Bilirubin"),
    gr.inputs.Number(label="Alkaline Phosphotase"),
    gr.inputs.Number(label="Aspartate Aminotransferase"),
    gr.inputs.Number(label="Albumin"),
    gr.inputs.Number(label="Albumin and Globulin Ratio")
]

# Define the output field for the Gradio interface
output = gr.outputs.Label()

# Create the Gradio interface with the predict_liver_disease function as the underlying model
gr_interface = gr.Interface(fn=predict_liver_disease, inputs=inputs, outputs=output)

# Launch the Gradio interface
gr_interface.launch(share=True)
