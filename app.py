from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
from routes.camera import VideoCamera
from routes.db import db, get_logs_sorted, delete_all_logs, delete_logs_by_date, delete_logs_by_label
from routes.auth import auth_bp, login_required
import os
import reverse_geocoder as rg
from dotenv import load_dotenv

load_dotenv()

PORT = os.environ.get('PORT')
current_location = "Unknown"

def get_current_location():
    return current_location

def coords_to_location(lat, lon):
    results = rg.search((lat, lon))  # [{'name': 'Seoul', 'cc': 'KR', 'admin1': 'Seoul', ...}]
    if results:
        city = results[0]['name']      # ex) 'Seoul'
        country_code = results[0]['cc'] # ex) 'KR'
        # 필요하면 나라 코드 → 나라 이름 매핑 가능 (KR → 대한민국)
        return f"{country_code} {city}"
    return "Unknown"

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')  # 세션을 위한 시크릿 키

# ✅ DB 연결
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get('DB_URI')  # .env에서 읽기
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.create_all()

# 블루프린트 등록
app.register_blueprint(auth_bp)

# 통합된 음주 탐지기 사용 (BlazeFace 필수)
camera = VideoCamera(location_callback=get_current_location)

@app.context_processor
def inject_user():
    """모든 템플릿에 사용자 정보 주입"""
    return dict(
        logged_in=session.get('logged_in', False),
        user_id=session.get('user_id'),
        nickname=session.get('nickname')
    )

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/detect')
@login_required
def detect():
    return render_template('detect.html')

@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')

@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log')
@login_required
def log():
    sort_by = request.args.get("by", "timestamp")
    order = request.args.get("order", "desc")
    drunk_users = get_logs_sorted(by=sort_by, order=order)
    return render_template('log.html', logs=drunk_users)

@app.route("/delete_logs/all", methods=["POST"])
def delete_all():
    delete_all_logs()
    return redirect(url_for("log"))

@app.route("/delete_logs/date", methods=["POST"])
def delete_by_date():
    date = request.form.get("date")
    if date:
        delete_logs_by_date(date)
    return redirect(url_for("log"))

@app.route("/delete_logs/label", methods=["POST"])
def delete_by_label():
    label = request.form.get("label")
    if label:
        delete_logs_by_label(label)
    return redirect(url_for("log"))

@app.route("/set_threshold", methods=["POST"])
def set_threshold():
    """임계값 동적 조절"""
    try:
        threshold = float(request.form.get("threshold", 0.7))
        if 0.0 <= threshold <= 1.0:
            camera.set_threshold(threshold)
            return jsonify({"success": True, "threshold": threshold, "message": f"임계값이 {threshold}로 변경되었습니다."})
        else:
            return jsonify({"success": False, "message": "임계값은 0.0과 1.0 사이여야 합니다."})
    except ValueError:
        return jsonify({"success": False, "message": "올바른 숫자를 입력해주세요."})

@app.route("/get_threshold", methods=["GET"])
def get_threshold():
    """현재 임계값 조회"""
    try:
        current_threshold = camera.get_current_threshold()
        return jsonify({"success": True, "threshold": current_threshold})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route("/submit_location", methods=["POST"])
def submit_location():
    global current_location
    data = request.get_json()
    lat, lon = data.get("latitude"), data.get("longitude")
    if lat and lon:
        try:
            lat, lon = float(lat), float(lon)
            current_location = coords_to_location(lat, lon)  # "KR Seoul"
        except Exception as e:
            print("Location error:", e)
            current_location = f"{lat},{lon}"
    return {"status": "success"}

if __name__ == '__main__':
    print(f"{PORT}번 포트에서 대기 중")
    app.run('0.0.0.0', port=PORT, debug=True)
