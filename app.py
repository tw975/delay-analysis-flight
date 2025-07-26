from flask import Flask, request, render_template
import numpy as np
import pickle

model = pickle.load(open('flight.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    month = int(request.form['month'])
    dayofmonth = int(request.form['dayofmonth'])
    dayofweek = int(request.form['dayofweek'])
    dept = int(request.form['dept'])     
    arr = int(request.form['arrtime'])    
    actdept = int(request.form['actdept'])   
    distance = int(request.form['distance'])

    dep15 = actdept - dept

    total = [[month, dayofmonth, dayofweek, dept,
              arr, actdept, dep15, distance]]

    y_pred = model.predict(total)

    if y_pred[0] == 0:
        ans = "The flight will be on time"
    else:
        ans = "The flight will be delayed"

    return render_template('index.html', showcase=ans)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
