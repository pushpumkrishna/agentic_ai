import json
from functools import wraps
from http import HTTPStatus
from flask import request, jsonify


# Decorator to validate payload
def validate_payload(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Load the request data and confirm it is a valid JSON
            data = json.loads(request.get_data())

            # Check if 'email', 'subject', and 'body' are present in the payload
            if (
                "email" not in data
                or "subject" not in data["email"]
                or "body" not in data["email"]
            ):
                return jsonify(
                    {
                        "subject": HTTPStatus.BAD_REQUEST,
                        "body": "missing 'subject' or 'body' in payload",
                    }
                )

            # Validate that 'subject' and 'body' are non-empty strings
            subject = data["email"]["subject"]
            body = data["email"]["body"]

            if not isinstance(subject, str) or not subject.strip():
                return jsonify(
                    {
                        "subject": HTTPStatus.BAD_REQUEST,
                        "body": "invalid or empty 'subject'",
                    }
                )

            if not isinstance(body, str) or not body.strip():
                return jsonify(
                    {
                        "subject": HTTPStatus.BAD_REQUEST,
                        "body": "invalid or empty 'body'",
                    }
                )
        except ValueError:
            # Return error if payload is not a valid JSON
            return jsonify(
                {"subject": HTTPStatus.BAD_REQUEST, "body": "invalid JSON payload"}
            )
        # Proceed to the wrapped function if all validations pass
        return func(*args, **kwargs)

    return wrapper
