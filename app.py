from flask import Flask,render_template,request
import pickle


scaler=pickle.load(open("Model/standardScalar.pkl","rb"))
model=pickle.load(open("Model/modelForPrediction.pkl","rb"))
app=Flask(__name__)


@app.route("/")

def hello():
    return render_template('index.html')
    


@app.route("/predict_datapoint",methods=["GET","POST"])
def predict_datapoint():
      if request.method=="POST":
            preg=int(request.form.get("Pregnancies"))
            glu=int(request.form.get("Glucose"))
            bp=int(request.form.get("BloodPressure"))
            skin=int(request.form.get("SkinThickness"))
            ins=int(request.form.get("Insulin"))
            bmi=int(request.form.get("BMI"))
            pd=int(request.form.get("DiabetesPedigreeFunction"))
            age=int(request.form.get("Age"))
            lists=[preg,glu,bp,skin,ins,bmi,pd,age]
            scaled=scaler.transform([lists])
            output=model.predict(scaled)
            if output[0]==1:
                  final="Diabetic"
                  page="single_prediction.html"
            else:
                  final="Non Diabetic"
                  page="single_prediction2.html"      
            return render_template(page,result=final)






      else:
           return render_template('home.html')

            
      



