# Import libraries
import sys

sys.path.insert(1, "/home/ubuntu/app/Bike-Sharing-Demand/dags/app")

from flask import Flask, render_template, request
from predict import predict

app = Flask(__name__)

@app.route("/")
def my_form():
    """
    Displays the web page, text and dropdown boxes
    """

    season = ["Spring", "Summer", "Autumn", "Winter"]
    holiday = ["No", "Yes"]
    workingday = ["No", "Yes"]
    weather = ["Clear, Few clouds, Partly cloudy, Partly cloudy","Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist", "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"]

    return render_template(
        "index.html",
        season=season,
        holiday=holiday,
        workingday=workingday,
        weather=weather
    )


@app.route("/", methods=["POST", "GET"])
def enter():
    """
    Receives user input on the web and computes the prediction
    """
    # temp	atemp	humidity	windspeed
    if request.method == "POST":
        temp = request.form['nm']
        atemp = request.form['rm']
        humidity = request.form['hm']
        windspeed = request.form['wm']
        season_val = request.form.get('season')
        holiday_val = request.form.get('holiday')
        workingday_val = request.form.get('workingday')
        weather_val = request.form.get('weather')
        results = {}
        results['temp'] = temp
        results['atemp'] = atemp
        results['humidity'] = humidity
        results['windspeed'] = windspeed
        results['season'] = season_val
        results['holiday'] = holiday_val
        results['workingday'] = workingday_val
        results['weather'] = weather_val
        prediction = predict(results)

        return render_template(
            "index.html",
            result=prediction,
            temp=temp,
            atemp=atemp,
            humidity=humidity,
            windspeed=windspeed,
            season_val=season_val,
            holiday_val=holiday_val,
            workingday_val=workingday_val,
            weather_val=weather_val,
        )

    return render_template("index.html")

        


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3500)
