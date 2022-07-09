# flask, pandas, scikit learn, pickle-mixin
# importing libraries


from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
# loading the dataset
car = pd.read_csv("cleaned_Car.csv")


@app.route("/")
def index():
    # Adding all the criteria need for prediction
    companies = sorted(car["manufacturer"].unique())
    conditions = sorted(car["condition"].unique())
    cylinders = sorted(car["cylinders"].unique())
    fuels = sorted(car["fuel"].unique())
    transmission = sorted(car["transmission"].astype(str).unique())
    drives = sorted(car["drive"].unique())
    types = sorted(car["type"].unique())
    car_models = sorted(car["model_updated"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    # Distance_Traveled

    return render_template("index.html", companies=companies,
                           car_models=car_models, conditions=conditions, cylinders=cylinders, fuels=fuels,
                           transmissions=transmission, drives=drives, types=types,
                           years=year)


# predicting the price
@app.route("/predict", methods=["POST"])
def predict():
    company = request.form.get("company")
    Car = request.form.get("car_models")
    condition = request.form.get("condition")
    cylinder = request.form.get("cylinder")
    fuel = request.form.get("fuel")
    transmission = request.form.get("transmission")
    Drive = request.form.get("drive")
    Type = request.form.get("type")
    year = int(request.form.get("year"))
    distance = int(request.form.get("distance"))
    print(company, Car, condition, cylinder, fuel, transmission, Drive, Type, year, distance)
    prediction = model.predict(pd.DataFrame(
        [[year, company, Car, condition, cylinder, fuel, distance, transmission, Drive, Type]],
        columns=["year", "manufacturer", "model_updated", "condition",
                 "cylinders", "fuel", "Distance_Traveled",
                 "transmission", "drive", "type"]
        ))
    return str(np.round(prediction[0], 2))


if __name__ == "__main__":
    app.run(debug=False)
