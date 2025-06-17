from flask import Flask, render_template, Response, request, redirect, url_for
from camera import VideoCamera
# from db import get_drunk_logs
from db import get_logs_sorted
from db import delete_all_logs, delete_logs_by_date, delete_logs_by_label
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
    sort_by = request.args.get("by", "timestamp")
    order = request.args.get("order", "desc")
    drunk_users = get_logs_sorted(by=sort_by, order=order)
    # drunk_users = get_drunk_logs()
    return render_template('log.html', logs=drunk_users)

@app.route("/delete_logs/all", methods=["POST"])
def delete_all():
    delete_all_logs()
    return redirect(url_for("log"))  # 로그 페이지로 리다이렉트

# 날짜 기준 삭제 요청
@app.route("/delete_logs/date", methods=["POST"])
def delete_by_date():
    date = request.form.get("date")
    if date:
        delete_logs_by_date(date)
    return redirect(url_for("log"))

# 라벨 기준 삭제 요청
@app.route("/delete_logs/label", methods=["POST"])
def delete_by_label():
    label = request.form.get("label")
    if label:
        delete_logs_by_label(label)
    return redirect(url_for("log"))

if __name__ == '__main__':
    print(f"{PORT}번 포트에서 대기 중")
    app.run('0.0.0.0', port=PORT, debug=True)