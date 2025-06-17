from flask import Flask, render_template, Response, request, redirect
from camera import VideoCamera
from db import get_drunk_logs
import os

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
    return Response(camera.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log')
def log():
    drunk_users = get_drunk_logs()
    return render_template('log.html', logs=drunk_users)

if __name__ == '__main__':
    print(f"{PORT}번 포트에서 대기 중")
    app.run('0.0.0.0', port=PORT, debug=True)