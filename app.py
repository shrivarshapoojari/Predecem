from flask import Flask , request , jsonify , render_template
import pickle
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)
model=pickle.load(open('models/randomforest.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        cement=int(request.form.get('cem1'))
        bfs=int(request.form.get('cem2'))
        fa=int(request.form.get('cem3'))
        water=int(request.form.get('cem4'))
        sup=int(request.form.get('cem5'))
        ca=int(request.form.get('cem6'))
        fine=int(request.form.get('cem7'))
        age=int(request.form.get('cem8'))
        
        new_data=[[cement,bfs,fa,water,sup,ca,fine,age]]
        scaled_data = standard_scaler.transform(new_data)
        result=model.predict(scaled_data)
        print(new_data)
        prediction_result =int( result[0])
        print(prediction_result)
        return render_template("input.html",strength=str(prediction_result)+" megapascals ")
    else:
        return  render_template("input.html")
          
    


if __name__=='main':
    app.run(debug=True)
