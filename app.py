from flask import Flask
from src.logger import logging
from src.exception import CustmeException  # This is correct if class is named CustmeException
import os, sys

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        raise Exception("We are testing our custom file")
    except Exception as e:
        # Handle the exception using your custom class
        abc = CustmeException(e, sys)  # This is correct if class is named CustmeException
        logging.error(f"An error occurred: {abc}")
        return "Welcome to My AQI Project of 10PEARLS"

if __name__ == "__main__":
    app.run(debug=True)