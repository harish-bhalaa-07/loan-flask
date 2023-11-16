import numpy as np
from flask import Flask,request,render_template
import pickle
import os

os.chdir('./')

app=Flask(__name__)
model=pickle.load(open('loan_prediction_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('loan_prediction.html')  #html page here


@app.route('/predict',methods=['POST'])
def predict():
    
    #storing the input values from html form
    features=[[int(x) for x in request.form.values()]]
    final_features=np.array(features)  
    
    print(final_features)

    #prediction=model.predict(final_features)
    prediction = model.predict(final_features)
    prediction=prediction.tolist()
    
    if prediction == [1]:
        prediction = "Loan will be provided"
    else:
        prediction = "Loan will not be provided"
        

    return render_template('loan_prediction.html', prediction_value = prediction)

if __name__=="__main__":
    app.run(debug=True)
    
