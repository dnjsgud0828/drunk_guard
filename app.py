from flask import Flask, render_template
from dotenv import load_dotenv
import os

#load .env
load_dotenv

PORT = os.environ.get('PORT')
KAKAO_ID = os.environ.get('KAKAO_ID')

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/elements')
def elements():
    return render_template('elements.html')

@app.route('/generic')
def generic():
    return render_template('generic.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signUp')
def signUp():
    return render_template('signUp.html')

if __name__ == '__main__':
    print(f"{PORT}번 포트에서 대기 중")
    app.run('0.0.0.0', port=PORT, debug=True)