from app import app

# This tells Gunicorn to find the Flask application instance named 'app'
# within the 'app' module (i.e., app.py).
if __name__ == "__main__":
    app.run()