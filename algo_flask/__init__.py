from flask import Flask
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ecf6e975838a2f7bf3c5dbe7d55ebe5b'
sys.path.append('/home/AndreyFridman0/algo_research')  # import algo_research module from server
from algo_flask import routes
