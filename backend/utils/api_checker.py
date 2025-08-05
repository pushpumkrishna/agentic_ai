from functools import wraps
from http import HTTPStatus
from flask import request, jsonify


# Wrapper to check if the incoming request method is OPTIONS
def api_checker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # If the request method is OPTIONS, return a response indicating the API is alive
        if request.method == "OPTIONS":
            return jsonify({"status": HTTPStatus.OK, "message": "API is alive"})
        # Otherwise, pass control to the decorated function
        return func(*args, **kwargs)

    return wrapper
