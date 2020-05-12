import pickle
#import cPickle
import numpy as np
from flask import Flask, render_template, request

from ML import model1

app=Flask(__name__)

model2=pickle.load(open("model2.pkl","rb"))
#print(model2([1143.900000],1))

model1=pickle.load(open('model1.pkl','rb'))
#print(model1([1143.900000],10))

model3=pickle.load(open("model3.pkl","rb"))
#print(model3([1143.900000],1))

model4=pickle.load(open("model4.pkl","rb"))
#print(model4([1143.900000],1))

#def unpickle(model1):
    #with open(model1.pkl) as f:
        #obj = cPickle.load(f)
    #return obj

@app.route("/")
def hello_world():
    return render_template("file.html")

@app.route("/predict",methods=["POST","GET"])
def predict():
    int_features=[int(x) for x in request.form.values()]

    final=np.array(int_features)
    state=final[0]
    m=final[1]
    rain=[final[2]]
    #last=(rain, state)

    if m == 11:
        predicted_data=model1(rain,state)
        print(predicted_data)
    elif m == 12:
        predicted_data=model2(rain,state)
        print(predicted_data)
    elif m == 13:
        predicted_data=model3(rain,state)
        print(predicted_data)
    elif m == 14:
        predicted_data=model4(rain,state)
        print(predicted_data)

    output = '{0:.{1}f}'.format(predicted_data[0], 3)
    return render_template('file.html', pred=' Prediction for annual rainfall is {} mm'.format(output))



if __name__=="__main__":
    app.run(debug=True)
