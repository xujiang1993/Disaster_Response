from flask import Flask

app = Flask(__name__)
from utils import tokenize
from DRapp import run