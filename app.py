#importing relevant libraries for flask,html rendering and loading the ML module
from distutils.log import debug
from flask import Flask, request, url_for,redirect,render_template
import pickle
import joblib
import pandas as pd
import xgboost


app= Flask(__name__)


# model = pickle.load(open('model.pkl','rb'))
model = joblib.load('model.pkl')

# scale = pickle.load(open('scale.pkl','rb'))
scale = joblib.load('scale.pkl')

@app.route('/')
def landing_page():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    Pregnancies = request.form['1']
    Glucose = request.form['2']
    BloodPressure = request.form['3']
    SkinThickness = request.form['4']
    Insulin = request.form['5']
    BMI = request.form['6']
    DPF = request.form['7']
    Age = request.form['8']

    rawDf = pd.DataFrame([pd.Series([Pregnancies,Glucose,
    BloodPressure,SkinThickness,Insulin,BMI,DPF,Age])])

    rawDf_new = pd.DataFrame(scale.transform(rawDf))
    
    print(rawDf_new)


    # model Prediction
    prediction = model.predict_proba(rawDf_new)
    print(f'The predicted value is: {prediction}')

    if prediction[0][1] >= 0.5:
        val_pred = round(prediction[0][1],4)
        return render_template('result.html',pred=f'You have a chance of having diabetes.\n probability of having diabetic is: {val_pred*100}% \n Excercise daily')
    else:
        val_pred = round(prediction[0][0],4)
        
        return render_template('result.html',pred=f'Congrats!!!!! you are safe.\n probability of being non- diabetic is:{val_pred*100}% \n You are in safe zone.\n\n workout daily to keep healthy and going')
        # return render_template('result.html',pred=f'Congrats!!!!! you are safe.\n probability of being non- diabetic is:{prediction[0][0]*100:.2f}% \n You are in safe zone.\n\n workout daily to keep healthy and going') .2f gives the number of decimals

if __name__=='__main__':
    app.run(debug=True)
    