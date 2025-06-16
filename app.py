from flask import Flask, render_template, Response, request, redirect
from camera import VideoCamera
from dotenv import load_dotenv
# from db import save_prediction, get_drunk_users
import os

load_dotenv

PORT = os.environ.get('PORT')

app = Flask(__name__)
camera = VideoCamera()

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera.generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log')
def log():
    # drunk_users = get_drunk_users()
    # return render_template('log.html', users=drunk_users)
    return render_template('log.html')

if __name__ == '__main__':
    print(f"{PORT}번 포트에서 대기 중")
    app.run('0.0.0.0', port=PORT, debug=True)