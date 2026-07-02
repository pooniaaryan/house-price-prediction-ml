from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(
    open("models/house_price_model.pkl", "rb")
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    TotalLivingArea = float(request.form["TotalLivingArea"])
    OverallQual = float(request.form["OverallQual"])
    BsmtQual = float(request.form["BsmtQual"])
    GrLivArea = float(request.form["GrLivArea"])
    BsmtFinSF1 = float(request.form["BsmtFinSF1"])
    YearRemodAdd = float(request.form["YearRemodAdd"])
    YearBuilt = float(request.form["YearBuilt"])
    HouseAge = float(request.form["HouseAge"])
    GarageArea = float(request.form["GarageArea"])
    LotArea = float(request.form["LotArea"])

    features = np.array([[
        TotalLivingArea,
        OverallQual,
        BsmtQual,
        GrLivArea,
        BsmtFinSF1,
        YearRemodAdd,
        YearBuilt,
        HouseAge,
        GarageArea,
        LotArea
    ]])

    prediction = model.predict(features)[0]

return render_template(
    "index.html",
    prediction=f"{prediction:,.2f}"
)

if __name__ == "__main__":
    app.run(debug=True)
