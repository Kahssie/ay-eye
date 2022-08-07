from flask import Flask
app = Flask(__name__)

# Initialize Flask app
app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug=True)
    
from core import home, modelprocessing