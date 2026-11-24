import os
import json
import random
import logging
from logging.handlers import RotatingFileHandler
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.ensemble import IsolationForest

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# --- Core config ---
app.config['SECRET_KEY'] = os.environ.get('NEUROGATE_SECRET_KEY', 'dev-secret-change-me')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///neurogate.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# IST timezone (UTC+5:30) — used to display timestamps in the user's local time.
IST = timezone(timedelta(hours=5, minutes=30))

@app.template_filter('to_ist')
def to_ist_filter(dt, fmt='%Y-%m-%d %H:%M:%S'):
    """Convert a naive-UTC datetime to IST for display in templates."""
    if dt is None:
        return '—'
    return dt.replace(tzinfo=timezone.utc).astimezone(IST).strftime(fmt)

# --- Real Disk Log Files Setup ---
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
ACCESS_LOG_FILE = os.path.join(LOGS_DIR, 'access.log')
SECURITY_LOG_FILE = os.path.join(LOGS_DIR, 'security.log')

class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone(IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%Y-%m-%d %H:%M:%S IST')

ist_formatter = ISTFormatter('[%(asctime)s] [%(levelname)s] %(message)s')

access_handler = RotatingFileHandler(ACCESS_LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
access_handler.setLevel(logging.INFO)
access_handler.setFormatter(ist_formatter)

security_handler = RotatingFileHandler(SECURITY_LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
security_handler.setLevel(logging.WARNING)
security_handler.setFormatter(ist_formatter)

app_logger = logging.getLogger('isogate')
app_logger.setLevel(logging.INFO)
if not app_logger.handlers:
    app_logger.addHandler(access_handler)
    app_logger.addHandler(security_handler)

# Blocks the brute-force path that the failed-attempt lockout alone doesn't
# fully cover: someone hammering /login or /send-otp fast from one IP.
limiter = Limiter(get_remote_address, app=app, default_limits=[])

# --- Email (EmailJS, sent client-side from the browser) ---
ADMIN_EMAIL = os.environ.get('NEUROGATE_ADMIN_EMAIL', 'tanvisuares2005@gmail.com')

ADMIN_EMAILS = set(
    e.strip().lower() for e in os.environ.get(
        'NEUROGATE_ADMIN_EMAILS',
        f'{ADMIN_EMAIL},vishmithasalian@gmail.com'
    ).split(',') if e.strip()
)

MAX_FAILED_ATTEMPTS = 4
OTP_TTL_MINUTES = 15

# email -> {"code": str, "expires": datetime}
otp_store = {}


def is_admin_email(email):
    """Check if an email has admin privileges (either configured or in DB)."""
    if not email:
        return False
    clean = email.strip().lower()
    if clean in ADMIN_EMAILS:
        return True
    u = User.query.filter_by(email=clean).first()
    return bool(u and u.is_admin)


def is_admin_session():
    """True only if the user logged in via /admin/login (session['admin_user'] is set)."""
    admin_user = session.get('admin_user')
    return bool(admin_user and is_admin_email(admin_user))


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    raw_password = db.Column(db.String(255), nullable=True)
    failed_attempts = db.Column(db.Integer, default=0)
    locked = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, raw):
        self.password_hash = generate_password_hash(raw)
        self.raw_password = raw

    def check_password(self, raw):
        return check_password_hash(self.password_hash, raw)


class AccessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120))
    event_type = db.Column(db.String(40))   # login, login_failed, account_locked, logout, file_access, admin_login, behavior_anomaly
    detail = db.Column(db.String(255))
    ip_address = db.Column(db.String(64))
    device = db.Column(db.String(255))
    location = db.Column(db.String(255))          # coarse, IP-based
    ip_lat = db.Column(db.Float, nullable=True)    # coarse coords, from IP geolocation
    ip_lng = db.Column(db.Float, nullable=True)
    precise_location = db.Column(db.String(255))   # exact, from browser geolocation (if granted)
    precise_lat = db.Column(db.Float, nullable=True)
    precise_lng = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class BehaviorBaseline(db.Model):
    """Rolling per-user baseline of typing/mouse rhythm, used to flag
    sessions that don't behave like the account's usual owner."""
    user_email = db.Column(db.String(120), primary_key=True)
    avg_flight = db.Column(db.Float, default=0.0)      # ms between keystrokes
    avg_velocity = db.Column(db.Float, default=0.0)    # px/ms mouse speed
    sample_count = db.Column(db.Integer, default=0)
    last_deviation = db.Column(db.Float, default=0.0)  # 0.0-1.0+, most recent sample's risk
    last_seen = db.Column(db.DateTime, nullable=True)
    last_method = db.Column(db.String(20), default='learning')  # learning | statistical | isolation_forest


class BehaviorSample(db.Model):
    """Every raw behavioral reading, kept so the Isolation Forest has a real
    history to train on and so the admin charts can show a trend over time."""
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), index=True)
    avg_flight = db.Column(db.Float)
    avg_velocity = db.Column(db.Float)
    total_keystrokes = db.Column(db.Integer)
    risk_score = db.Column(db.Float)   # 0-100
    method = db.Column(db.String(20))  # learning | statistical | isolation_forest
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()
    # Add raw_password column to SQLite table if it's missing from existing DB file
    try:
        db.session.execute(text("ALTER TABLE user ADD COLUMN raw_password VARCHAR(255)"))
        db.session.commit()
    except Exception:
        db.session.rollback()

    # Seed demo accounts the first time the DB is created.
    if not User.query.first():
        demo_users = [
            ('norman@gmail.com', 'nor123', False),
            (ADMIN_EMAIL, 'admin123', True),
            ('vishmithasalian@gmail.com', 'admin123', True),
        ]
        for email, pw, is_admin in demo_users:
            u = User(email=email, is_admin=is_admin)
            u.set_password(pw)
            db.session.add(u)
        db.session.commit()
    else:
        # Populate admin status and raw_password if missing.
        for u in User.query.all():
            if u.email.lower() in ADMIN_EMAILS:
                u.is_admin = True
            if not u.raw_password:
                if u.email == 'norman@gmail.com':
                    u.raw_password = 'nor123'
                else:
                    u.raw_password = 'admin123'
        db.session.commit()


def generate_otp():
    return f"{random.randint(0, 999999):06d}"


def get_client_ip():
    """Extract real client IP even behind reverse proxies or tunnels."""
    if request:
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            return forwarded.split(',')[0].strip()
        return request.remote_addr or '127.0.0.1'
    return '127.0.0.1'


def parse_device(ua_string):
    """Lightweight User-Agent parsing — no extra dependency needed."""
    ua = (ua_string or '').lower()
    if 'ipad' in ua or 'tablet' in ua:
        device_type = 'Tablet'
    elif 'mobile' in ua or 'android' in ua or 'iphone' in ua:
        device_type = 'Mobile'
    else:
        device_type = 'Desktop'

    if 'windows' in ua:
        os_name = 'Windows'
    elif 'mac os' in ua or 'macintosh' in ua:
        os_name = 'macOS'
    elif 'android' in ua:
        os_name = 'Android'
    elif 'iphone' in ua or 'ipad' in ua:
        os_name = 'iOS'
    elif 'linux' in ua:
        os_name = 'Linux'
    else:
        os_name = 'Unknown OS'

    if 'edg/' in ua:
        browser = 'Edge'
    elif 'chrome/' in ua and 'chromium' not in ua:
        browser = 'Chrome'
    elif 'firefox/' in ua:
        browser = 'Firefox'
    elif 'safari/' in ua and 'chrome' not in ua:
        browser = 'Safari'
    else:
        browser = 'Unknown browser'

    return f"{browser} on {os_name} ({device_type})"


def get_location(ip):
    """Free, robust IP geolocation via ip-api.com with freeipapi fallback.
    Returns (label, lat, lng)."""
    is_local = not ip or ip in ('127.0.0.1', 'localhost', '::1') or ip.startswith(('192.168.', '10.', '172.'))
    lookup_url = ("http://ip-api.com/json/?fields=status,city,regionName,country,lat,lon"
                  if is_local else
                  f"http://ip-api.com/json/{ip}?fields=status,city,regionName,country,lat,lon")
    try:
        req = urllib.request.Request(lookup_url, headers={'User-Agent': 'IsoGate-Security/1.0'})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
        if data.get('status') == 'success':
            parts = [p for p in (data.get('city'), data.get('regionName'), data.get('country')) if p]
            label = ', '.join(parts) if parts else 'Unknown'
            if is_local:
                label += ' (approx., local network)'
            lat, lon = data.get('lat'), data.get('lon')
            if lat is not None and lon is not None:
                return label, float(lat), float(lon)
    except Exception:
        pass

    # Secondary provider fallback: freeipapi.com
    try:
        fb_url = "https://freeipapi.com/api/json" if is_local else f"https://freeipapi.com/api/json/{ip}"
        req = urllib.request.Request(fb_url, headers={'User-Agent': 'IsoGate-Security/1.0'})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
        city = data.get('cityName')
        region = data.get('regionName')
        country = data.get('countryName')
        lat = data.get('latitude')
        lon = data.get('longitude')
        if lat is not None and lon is not None:
            parts = [p for p in (city, region, country) if p]
            label = ', '.join(parts) if parts else 'Local network'
            if is_local:
                label += ' (approx., local network)'
            return label, float(lat), float(lon)
    except Exception:
        pass

    # Fallback coordinates for local testing so Map is never blank
    if is_local:
        return 'Local network (approx. Mangalore, IN)', 12.9141, 74.8560
    return 'Unknown', None, None


def reverse_geocode(lat, lng):
    """Turns exact browser-reported coordinates into a readable place name,
    via OpenStreetMap's free Nominatim service."""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lng}&format=json&zoom=16"
        req = urllib.request.Request(url, headers={'User-Agent': 'IsoGate-College-Project/1.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        addr = data.get('address', {})
        parts = [
            addr.get('road') or addr.get('suburb') or addr.get('neighbourhood'),
            addr.get('city') or addr.get('town') or addr.get('village'),
            addr.get('state'),
        ]
        parts = [p for p in parts if p]
        return ', '.join(parts) if parts else data.get('display_name', f"{lat}, {lng}")
    except Exception:
        return f"{lat}, {lng}"


ISO_MIN_SAMPLES = 8  # need this many past samples before the ML model kicks in


def score_behavior(email, flight, velocity):
    """Real anomaly scoring: fits a fresh Isolation Forest on this user's own
    past samples (their normal typing/mouse rhythm) and scores the new one
    against it. Below ISO_MIN_SAMPLES history, falls back to a simple
    percentage-deviation check so new users aren't flagged pointlessly on
    day one. Returns (risk_score 0-100, method, is_anomaly)."""
    history = (BehaviorSample.query
               .filter_by(user_email=email)
               .order_by(BehaviorSample.timestamp.desc())
               .limit(60).all())

    if len(history) >= ISO_MIN_SAMPLES:
        X = np.array([[h.avg_flight, h.avg_velocity] for h in history])
        model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
        model.fit(X)
        train_scores = model.decision_function(X)
        new_score = model.decision_function([[flight, velocity]])[0]

        lo, hi = train_scores.min(), train_scores.max()
        risk = 100 * (hi - new_score) / (hi - lo + 1e-9)
        risk = float(max(0.0, min(100.0, risk)))
        return risk, 'isolation_forest', risk > 60

    baseline = BehaviorBaseline.query.get(email)
    if baseline and baseline.sample_count >= 3:
        flight_dev = abs(flight - baseline.avg_flight) / (baseline.avg_flight or 1)
        vel_dev = abs(velocity - baseline.avg_velocity) / (baseline.avg_velocity or 1)
        risk = float(max(0.0, min(100.0, max(flight_dev, vel_dev) * 100)))
        return risk, 'statistical', risk > 60

    return 0.0, 'learning', False


def current_email():
    """A signed-in identity, whichever door they came through — the
    regular /login session or the /admin/login session."""
    return session.get('user') or session.get('admin_user')


def log_event(email, event_type, detail='', ip=None, device=None,
              location=None, ip_lat=None, ip_lng=None,
              precise_location=None, precise_lat=None, precise_lng=None):
    """Logs security and access events to both SQLite AccessLog and real rotating log files."""
    if not ip:
        ip = get_client_ip()
    if not device:
        device = parse_device(request.headers.get('User-Agent') if request else '')
    if location is None:
        label, calc_lat, calc_lng = get_location(ip)
        if ip_lat is None:
            ip_lat = calc_lat
        if ip_lng is None:
            ip_lng = calc_lng
    else:
        label = location

    entry = AccessLog(
        user_email=email,
        event_type=event_type,
        detail=detail,
        ip_address=ip,
        device=device,
        location=label,
        ip_lat=ip_lat,
        ip_lng=ip_lng,
        precise_location=precise_location,
        precise_lat=precise_lat,
        precise_lng=precise_lng,
    )
    db.session.add(entry)
    db.session.commit()

    # Write structured entry to disk log files (logs/access.log & logs/security.log)
    loc_desc = precise_location or label
    log_line = (f"[{event_type.upper()}] user={email} ip={ip} "
                f"location=\"{loc_desc}\" device=\"{device}\" detail=\"{detail}\"")
    if event_type in ('account_locked', 'behavior_anomaly', 'login_failed'):
        app_logger.warning(log_line)
    else:
        app_logger.info(log_line)


# Dummy "organization" files — swap this out for real records/DB rows later.
# Grouped into categories so /files can show tabs instead of one flat grid.
FILE_CATEGORIES = {
    'Finance': [
        {'id': 1, 'name': 'Q3_Financial_Report.xlsx', 'dept': 'Finance', 'size': '842 KB',
         'preview': 'Quarterly revenue up 12% QoQ. Full regional breakdown in the real workbook.'},
        {'id': 2, 'name': 'Fee_Structure_2026.pdf', 'dept': 'Finance', 'size': '410 KB',
         'preview': 'Semester-wise fee breakdown by branch and category.'},
        {'id': 3, 'name': 'Scholarship_Disbursement.csv', 'dept': 'Finance', 'size': '96 KB',
         'preview': 'Scholarship payout records for the current academic year.'},
    ],
    'Student Documentation': [
        {'id': 4, 'name': 'Student_Handbook.pdf', 'dept': 'Academics', 'size': '1.2 MB',
         'preview': 'Institution policies, code of conduct, and academic regulations.'},
        {'id': 5, 'name': 'Hostel_Rules.pdf', 'dept': 'Academics', 'size': '310 KB',
         'preview': 'Hostel allotment process and residence guidelines.'},
        {'id': 6, 'name': 'Bonafide_Certificate_Template.docx', 'dept': 'Academics', 'size': '58 KB',
         'preview': 'Standard bonafide certificate format for student use.'},
    ],
    'Timetables': [
        {'id': 7, 'name': 'CCE_Timetable_Sem5.pdf', 'dept': 'CCE', 'size': '210 KB',
         'preview': 'Computer & Communication Engineering — Semester 5 class schedule.'},
        {'id': 8, 'name': 'CSE_Timetable_Sem5.pdf', 'dept': 'CSE', 'size': '208 KB',
         'preview': 'Computer Science & Engineering — Semester 5 class schedule.'},
        {'id': 9, 'name': 'AIML_Timetable_Sem5.pdf', 'dept': 'AIML', 'size': '215 KB',
         'preview': 'AI & Machine Learning — Semester 5 class schedule.'},
        {'id': 10, 'name': 'AIDS_Timetable_Sem5.pdf', 'dept': 'AIDS', 'size': '212 KB',
         'preview': 'AI & Data Science — Semester 5 class schedule.'},
    ],
    'Attendance': [
        {'id': 11, 'name': 'Student_Attendance_Aug2026.xlsx', 'dept': 'Academics', 'size': '640 KB',
         'preview': 'Monthly attendance sheet, branch-wise, for all registered students.'},
        {'id': 12, 'name': 'Lecturer_Attendance_Aug2026.xlsx', 'dept': 'Academics', 'size': '188 KB',
         'preview': 'Faculty attendance and substitution log for the month.'},
    ],
    'Registration': [
        {'id': 13, 'name': 'NNM23CC059_Registration.pdf', 'dept': 'CCE', 'size': '64 KB',
         'preview': 'Registration record for roll number NNM23CC059.'},
        {'id': 14, 'name': 'NNM23CC064_Registration.pdf', 'dept': 'CCE', 'size': '64 KB',
         'preview': 'Registration record for roll number NNM23CC064.'},
        {'id': 15, 'name': 'NNM23CC070_Registration.pdf', 'dept': 'CCE', 'size': '64 KB',
         'preview': 'Registration record for roll number NNM23CC070.'},
    ],
    'Mixed': [
        {'id': 16, 'name': 'Org_Chart_2026.png', 'dept': 'Operations', 'size': '310 KB',
         'preview': 'Current reporting structure across all departments.'},
        {'id': 17, 'name': 'Event_Calendar_2026.pdf', 'dept': 'Operations', 'size': '270 KB',
         'preview': 'Academic and cultural events calendar for the year.'},
        {'id': 18, 'name': 'IT_Helpdesk_Contacts.csv', 'dept': 'IT', 'size': '18 KB',
         'preview': 'Contact list for campus IT support.'},
    ],
}
DUMMY_FILES = [f for group in FILE_CATEGORIES.values() for f in group]


# ---------------- routes ----------------

@app.route('/')
def index():
    if session.get('admin_user'):
        return redirect(url_for('admin'))
    if session.get('user'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


def handle_login_attempt(template_name, success_endpoint, require_admin=False):
    """Shared logic for /login and /admin/login: lockout rules, OTP reset,
    admin alerts, and role verification."""
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''
    user = User.query.filter_by(email=email).first()
    generic_error = "Invalid email or password."

    lat_val = request.form.get('lat')
    lng_val = request.form.get('lng')
    c_lat, c_lng, c_place = None, None, None
    if lat_val and lng_val:
        try:
            c_lat = float(lat_val)
            c_lng = float(lng_val)
            c_place = reverse_geocode(c_lat, c_lng)
        except Exception:
            pass

    if not user:
        return render_template(template_name, error=generic_error, email=email)

    if require_admin and not is_admin_email(email):
        return render_template(template_name, error="This account is not authorized for admin access.", email=email)

    if user.locked:
        return render_template(
            template_name,
            error="Account locked due to repeated failed attempts. Reset your password to continue.",
            email=email, show_reset=True,
        )

    if user.check_password(password):
        user.failed_attempts = 0
        db.session.commit()
        if require_admin:
            # /admin/login path: set admin_user session only
            session['admin_user'] = user.email
            log_event(user.email, 'admin_login', 'Signed in to admin console',
                      precise_location=c_place, precise_lat=c_lat, precise_lng=c_lng)
        else:
            # /login path: always set user session only (no admin escalation)
            session['user'] = user.email
            log_event(user.email, 'login', 'Signed in successfully',
                      precise_location=c_place, precise_lat=c_lat, precise_lng=c_lng)
        return redirect(url_for(success_endpoint))

    # Wrong password
    user.failed_attempts += 1
    if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
        user.locked = True
        db.session.commit()
        alert_message = (
            f"{user.email} was locked out after {user.failed_attempts} failed login "
            f"attempts at {datetime.now(IST).strftime('%Y-%m-%d %I:%M %p IST')} "
            f"(IP: {get_client_ip()})."
        )
        log_event(user.email, 'account_locked', alert_message)
        return render_template(
            template_name,
            error="Too many failed attempts. Account locked and admin notified.",
            email=email, show_reset=True,
            trigger_admin_alert=True,
            admin_email=ADMIN_EMAIL,
            alert_message=alert_message,
        )

    db.session.commit()
    log_event(user.email, 'login_failed', f'Attempt {user.failed_attempts}/{MAX_FAILED_ATTEMPTS}')
    remaining = MAX_FAILED_ATTEMPTS - user.failed_attempts
    return render_template(
        template_name,
        error=f"{generic_error} ({remaining} attempt{'s' if remaining != 1 else ''} left)",
        email=email,
    )


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if request.method == 'GET':
        if session.get('user'):
            return redirect(url_for('dashboard'))
        return render_template('login.html')
    return handle_login_attempt('login.html', 'dashboard', require_admin=False)


@app.route('/admin/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def admin_login():
    if request.method == 'GET':
        if is_admin_session():
            return redirect(url_for('admin'))
        return render_template('admin_login.html')
    return handle_login_attempt('admin_login.html', 'admin', require_admin=True)


@app.route('/logout')
def logout():
    email = current_email()
    is_admin = is_admin_email(email)
    if email:
        log_event(email, 'logout', 'Signed out')
    session.clear()
    if is_admin:
        return redirect(url_for('admin_login'))
    return redirect(url_for('login'))


@app.route('/send-otp', methods=['POST'])
@limiter.limit("5 per minute")
def send_otp():
    data = request.get_json(silent=True) or request.form
    email = (data.get('email') or '').strip().lower()
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'status': 'error', 'error': 'No account registered with this email address.'}), 404

    code = generate_otp()
    expires_dt = datetime.utcnow() + timedelta(minutes=OTP_TTL_MINUTES)
    otp_store[email] = {'code': code, 'expires': expires_dt}
    expires_str = expires_dt.replace(tzinfo=timezone.utc).astimezone(IST).strftime('%I:%M %p IST')

    log_event(email, 'otp_requested', f"Generated 6-digit OTP valid for {OTP_TTL_MINUTES} mins until {expires_str}")

    return jsonify({
        'status': 'success',
        'otp': code,
        'email': email,
        'valid_until': expires_str,
        'time': expires_str,
        'expiry': expires_str,
        'expiry_time': expires_str,
        'until': expires_str,
        'expires': expires_str,
    })


@app.route('/reset-password', methods=['POST'])
def reset_password():
    email = (request.form.get('email') or '').strip().lower()
    user_otp = (request.form.get('user_otp') or '').strip()
    new_password = request.form.get('new_password') or ''
    source = request.form.get('source') or ''

    is_admin_req = (source == 'admin') or is_admin_email(email)
    target_template = 'admin_login.html' if is_admin_req else 'login.html'

    record = otp_store.get(email)
    if not record or datetime.utcnow() > record['expires'] or user_otp != record['code']:
        return render_template(target_template, error="Invalid or expired code.", email=email, show_reset=True)

    user = User.query.filter_by(email=email).first()
    if not user:
        return render_template(target_template, error="User not found.")

    if len(new_password) < 6:
        return render_template(target_template, error="Password must be at least 6 characters.",
                                email=email, show_reset=True)

    user.set_password(new_password)
    user.failed_attempts = 0
    user.locked = False
    db.session.commit()
    otp_store.pop(email, None)

    log_event(email, 'password_reset', 'Password successfully reset via OTP verification')

    return render_template(target_template, message="Password reset successfully. Please log in.")


@app.route('/dashboard')
def dashboard():
    email = session.get('user')
    if not email:
        return redirect(url_for('login'))

    session.pop('suspended', None)
    session.pop('suspended_reason', None)

    last_log = (AccessLog.query.filter_by(user_email=email)
                .order_by(AccessLog.timestamp.desc()).first())
    curr_loc = None
    if last_log:
        curr_loc = last_log.precise_location or last_log.location

    return render_template('dashboard.html', email=email, trust=0.82,
                            is_admin=False,
                            current_location=curr_loc)


@app.route('/files')
def files():
    email = session.get('user')
    if not email:
        return redirect(url_for('login'))

    return render_template('files.html', categories=FILE_CATEGORIES,
                            is_admin=False)


@app.route('/files/access/<int:file_id>', methods=['POST'])
def files_access(file_id):
    email = current_email()
    if not email:
        return jsonify({'error': 'Not authenticated'}), 401
    file = next((f for f in DUMMY_FILES if f['id'] == file_id), None)
    if not file:
        return jsonify({'error': 'File not found'}), 404
    log_event(email, 'file_access', file['name'])
    return jsonify({'name': file['name'], 'preview': file['preview']})


@app.route('/api/behavior', methods=['POST'])
def api_behavior():
    """Client posts rolling typing/mouse stats every ~20s. Scored against
    that user's own history via Isolation Forest — a big jump usually means
    someone other than the account owner is at the keyboard (e.g. a stolen
    session)."""
    if not session.get('user') and not session.get('admin_user'):
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json(silent=True) or {}
    flight = float(data.get('avg_typing_flight_time_ms') or 0)
    velocity = float(data.get('avg_mouse_velocity') or 0)
    keystrokes = int(data.get('total_keystrokes') or 0)
    mouse_samples = int(data.get('mouse_samples') or 0)
    email = current_email()

    # Typing rhythm OR mouse movement is enough signal — someone who only
    # moves the mouse (no typing) used to never generate a single sample,
    # which is why the chart could stay empty for an entire session.
    if keystrokes < 5 and mouse_samples < 5:
        return jsonify({'status': 'skipped'})  # not enough signal yet

    risk, method, anomaly = score_behavior(email, flight, velocity)

    db.session.add(BehaviorSample(
        user_email=email, avg_flight=flight, avg_velocity=velocity,
        total_keystrokes=keystrokes, risk_score=risk, method=method,
    ))

    # Rolling baseline — kept for the summary table and as the statistical
    # fallback's data source.
    alpha = 0.3
    baseline = BehaviorBaseline.query.get(email)
    if not baseline:
        baseline = BehaviorBaseline(user_email=email, avg_flight=flight, avg_velocity=velocity,
                                     sample_count=1, last_deviation=risk / 100, last_seen=datetime.utcnow(),
                                     last_method=method)
        db.session.add(baseline)
    else:
        baseline.avg_flight = baseline.avg_flight * (1 - alpha) + flight * alpha
        baseline.avg_velocity = baseline.avg_velocity * (1 - alpha) + velocity * alpha
        baseline.sample_count += 1
        baseline.last_deviation = risk / 100
        baseline.last_seen = datetime.utcnow()
        baseline.last_method = method
    db.session.commit()

    if anomaly or risk > 30:
        status_type = 'anomaly' if anomaly else 'elevated'
        reason = (f"Typing/mouse rhythm for {email} scored {round(risk)}% risk "
                   f"({method.replace('_', ' ')} model) — deviates from normal baseline.")
        log_event(email, 'behavior_anomaly', reason)
        return jsonify({
            'status': status_type,
            'elevated': True,
            'message': reason,
            'admin_email': ADMIN_EMAIL,
            'risk': round(risk, 1),
            'method': method
        })
    return jsonify({'status': 'ok', 'elevated': False, 'risk': round(risk, 1), 'method': method})


@app.route('/api/verify-password', methods=['POST'])
@limiter.limit("10 per minute")
def api_verify_password():
    email = current_email()
    if not email:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json(silent=True) or {}
    password = data.get('password') or ''
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    if user.locked:
        return jsonify({'error': 'Account locked.'}), 403
        
    if user.check_password(password):
        user.failed_attempts = 0
        session.pop('suspended', None)
        session.pop('suspended_reason', None)
        db.session.commit()
        
        # Prune recent anomalous samples so baseline isn't contaminated
        two_mins_ago = datetime.utcnow() - timedelta(minutes=2)
        BehaviorSample.query.filter(
            BehaviorSample.user_email == email,
            BehaviorSample.timestamp >= two_mins_ago,
            BehaviorSample.risk_score > 30
        ).delete()
        db.session.commit()
        
        log_event(email, 'behavior_verified', 'Behavior challenge solved via Password')
        return jsonify({'status': 'success'})
        
    user.failed_attempts += 1
    db.session.commit()
    
    if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
        user.locked = True
        db.session.commit()
        alert_message = (
            f"{user.email} was locked out after failing behavioral verification challenge "
            f"at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} (IP: {request.remote_addr})."
        )
        log_event(user.email, 'account_locked', alert_message)
        return jsonify({
            'status': 'locked',
            'error': 'Account locked due to too many failed attempts.',
            'admin_email': ADMIN_EMAIL,
            'alert_message': alert_message
        }), 403
        
    remaining = MAX_FAILED_ATTEMPTS - user.failed_attempts
    return jsonify({
        'status': 'error',
        'error': f"Incorrect password. {remaining} attempt{'s' if remaining != 1 else ''} left."
    }), 400


@app.route('/api/verify-otp', methods=['POST'])
@limiter.limit("10 per minute")
def api_verify_otp():
    email = current_email()
    if not email:
        return jsonify({'error': 'Not authenticated'}), 401
        
    data = request.get_json(silent=True) or {}
    code = (data.get('otp') or '').strip()
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    if user.locked:
        return jsonify({'error': 'Account locked.'}), 403
        
    record = otp_store.get(email)
    if not record or datetime.utcnow() > record['expires'] or code != record['code']:
        user.failed_attempts += 1
        db.session.commit()
        
        if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
            user.locked = True
            db.session.commit()
            alert_message = (
                f"{user.email} was locked out after failing behavioral verification challenge "
                f"at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} (IP: {request.remote_addr})."
            )
            log_event(user.email, 'account_locked', alert_message)
            return jsonify({
                'status': 'locked',
                'error': 'Account locked due to too many failed attempts.',
                'admin_email': ADMIN_EMAIL,
                'alert_message': alert_message
            }), 403
            
        remaining = MAX_FAILED_ATTEMPTS - user.failed_attempts
        return jsonify({
            'status': 'error',
            'error': f"Invalid or expired OTP code. {remaining} attempt{'s' if remaining != 1 else ''} left."
        }), 400
        
    otp_store.pop(email, None)
    user.failed_attempts = 0
    session.pop('suspended', None)
    session.pop('suspended_reason', None)
    db.session.commit()
    
    # Prune recent anomalous samples so baseline isn't contaminated
    two_mins_ago = datetime.utcnow() - timedelta(minutes=2)
    BehaviorSample.query.filter(
        BehaviorSample.user_email == email,
        BehaviorSample.timestamp >= two_mins_ago,
        BehaviorSample.risk_score > 30
    ).delete()
    db.session.commit()
    
    log_event(email, 'behavior_verified', 'Behavior challenge solved via OTP')
    return jsonify({'status': 'success'})


@app.route('/api/behavior/history')
def api_behavior_history():
    """Own recent samples, for the Chart.js graph on the dashboard."""
    email = current_email()
    if not email:
        return jsonify({'error': 'Not authenticated'}), 401
    samples = (BehaviorSample.query
               .filter_by(user_email=email)
               .order_by(BehaviorSample.timestamp.asc())
               .limit(50).all())
    return jsonify([{
        'timestamp': to_ist_filter(s.timestamp, '%H:%M:%S'),
        'risk_score': round(s.risk_score, 1),
        'avg_flight': s.avg_flight,
        'avg_velocity': s.avg_velocity,
        'method': s.method,
    } for s in samples])


@app.route('/api/location', methods=['POST'])
def api_location():
    """Browser sends exact GPS coordinates (after user grants permission).
    Reverse-geocodes and attaches it to the latest session login row."""
    email = current_email()
    if not email:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json(silent=True) or {}
    lat, lng = data.get('lat'), data.get('lng')
    accuracy = data.get('accuracy')
    if lat is None or lng is None:
        return jsonify({'error': 'Missing coordinates'}), 400

    try:
        lat = float(lat)
        lng = float(lng)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid coordinates'}), 400

    place = reverse_geocode(lat, lng)

    entry = (AccessLog.query
             .filter_by(user_email=email)
             .filter(AccessLog.event_type.in_(['login', 'admin_login']))
             .order_by(AccessLog.timestamp.desc())
             .first())
    if not entry:
        entry = (AccessLog.query
                 .filter_by(user_email=email)
                 .order_by(AccessLog.timestamp.desc())
                 .first())
    if entry:
        entry.precise_location = place
        entry.precise_lat = lat
        entry.precise_lng = lng
        if not entry.location or 'Unknown' in entry.location or 'Local network' in entry.location:
            entry.location = place
        db.session.commit()

    acc_text = f" (±{round(float(accuracy))}m)" if accuracy else ""
    log_event(email, 'location_verified', f"GPS Verified: {place}{acc_text}",
              location=place, precise_location=place, precise_lat=lat, precise_lng=lng)

    return jsonify({'status': 'success', 'place': place, 'lat': lat, 'lng': lng})


@app.route('/admin')
def admin():
    if not is_admin_session():
        return redirect(url_for('admin_login'))
    users = User.query.all()
    last_files = {}
    for u in users:
        last = (AccessLog.query
                .filter_by(user_email=u.email, event_type='file_access')
                .order_by(AccessLog.timestamp.desc())
                .first())
        last_files[u.email] = last.detail if last else None
    return render_template('admin.html', users=users, max_attempts=MAX_FAILED_ATTEMPTS,
                            last_files=last_files)


@app.route('/admin/activity')
def admin_activity():
    if not is_admin_session():
        return redirect(url_for('admin_login'))
    logs = AccessLog.query.order_by(AccessLog.timestamp.desc()).limit(200).all()
    return render_template('admin_activity.html', logs=logs)


@app.route('/admin/behavior')
def admin_behavior():
    if not is_admin_session():
        return redirect(url_for('admin_login'))
    baselines = BehaviorBaseline.query.order_by(BehaviorBaseline.last_seen.desc()).all()
    return render_template('admin_behavior.html', baselines=baselines)


@app.route('/api/admin/behavior')
def api_admin_behavior():
    """Polled by admin_behavior.html every few seconds for a live view."""
    if not is_admin_session():
        return jsonify({'error': 'Not authenticated'}), 401
    baselines = BehaviorBaseline.query.order_by(BehaviorBaseline.last_seen.desc()).all()
    return jsonify([{
        'email': b.user_email,
        'sample_count': b.sample_count,
        'deviation': round((b.last_deviation or 0) * 100, 1),
        'method': b.last_method or 'learning',
        'last_seen': to_ist_filter(b.last_seen) if b.last_seen else None,
        'status': (
            'learning' if b.sample_count < 3 else
            'anomaly' if (b.last_deviation or 0) > 0.6 else
            'elevated' if (b.last_deviation or 0) > 0.3 else
            'normal'
        ),
    } for b in baselines])


@app.route('/api/admin/behavior/history/<path:email>')
def api_admin_behavior_history(email):
    """A specific user's risk-score history, for the admin's Chart.js graph."""
    if not is_admin_session():
        return jsonify({'error': 'Not authenticated'}), 401
    samples = (BehaviorSample.query
               .filter_by(user_email=email)
               .order_by(BehaviorSample.timestamp.asc())
               .limit(60).all())
    return jsonify([{
        'timestamp': to_ist_filter(s.timestamp, '%H:%M:%S'),
        'risk_score': round(s.risk_score, 1),
        'method': s.method,
    } for s in samples])


@app.route('/admin/map')
def admin_map():
    if not is_admin_session():
        return redirect(url_for('admin_login'))
    return render_template('admin_map.html')


@app.route('/admin/database')
def admin_database():
    if not is_admin_session():
        return redirect(url_for('admin_login'))
    users = User.query.order_by(User.id).all()
    logs = AccessLog.query.order_by(AccessLog.timestamp.desc()).limit(500).all()
    baselines = BehaviorBaseline.query.order_by(BehaviorBaseline.last_seen.desc()).all()
    samples = BehaviorSample.query.order_by(BehaviorSample.timestamp.desc()).limit(500).all()
    return render_template('admin_database.html',
                           users=users, logs=logs, baselines=baselines, samples=samples)


@app.route('/api/admin/locations')
def api_admin_locations():
    """Recent user locations for the Leaflet Map — prioritizes precise GPS points."""
    if not is_admin_session():
        return jsonify({'error': 'Not authenticated'}), 401
    logs = (AccessLog.query
            .filter(AccessLog.event_type.in_(['login', 'admin_login', 'location_verified']))
            .order_by(AccessLog.timestamp.desc())
            .limit(200).all())

    user_locations = {}
    for log in logs:
        email = log.user_email
        if not email:
            continue
        lat = log.precise_lat if log.precise_lat is not None else log.ip_lat
        lng = log.precise_lng if log.precise_lng is not None else log.ip_lng
        if lat is None or lng is None:
            continue
        is_precise = (log.precise_lat is not None)
        label = log.precise_location or log.location or 'Logged in user'

        if email not in user_locations or (is_precise and not user_locations[email]['precise']):
            user_locations[email] = {
                'email': email,
                'lat': lat,
                'lng': lng,
                'precise': is_precise,
                'label': label,
                'timestamp': to_ist_filter(log.timestamp),
            }

    return jsonify(list(user_locations.values()))


@app.route('/admin/unlock/<int:user_id>', methods=['POST'])
def admin_unlock(user_id):
    if not is_admin_session():
        return redirect(url_for('admin_login'))
    user = User.query.get_or_404(user_id)
    user.locked = False
    user.failed_attempts = 0
    db.session.commit()
    log_event(current_email(), 'admin_unlock', f"Unlocked account for {user.email}")
    return redirect(url_for('admin'))


@app.route('/admin/toggle-role/<int:user_id>', methods=['POST'])
def admin_toggle_role(user_id):
    """Promote or demote user to/from Administrator role."""
    if not is_admin_session():
        return redirect(url_for('admin_login'))
    user = User.query.get_or_404(user_id)
    if user.email.lower() == current_email().lower():
        return redirect(url_for('admin'))  # Cannot demote self
    user.is_admin = not user.is_admin
    db.session.commit()
    new_role = "Administrator" if user.is_admin else "Regular User"
    log_event(current_email(), 'role_change', f"Changed {user.email} role to {new_role}")
    return redirect(url_for('admin'))


@app.route('/admin/logs/download/<log_type>')
def admin_download_log(log_type):
    """Download rotating plain-text disk log file."""
    if not is_admin_session():
        return redirect(url_for('admin_login'))
    if log_type == 'security':
        filepath = SECURITY_LOG_FILE
        download_name = 'security.log'
    else:
        filepath = ACCESS_LOG_FILE
        download_name = 'access.log'

    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# IsoGate {download_name} initialized at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}\n")

    return send_file(filepath, as_attachment=True, download_name=download_name, mimetype='text/plain')


@app.route('/api/logs')
def api_logs():
    email = current_email()
    if not email:
        return jsonify([])
    
    logs = AccessLog.query.filter_by(user_email=email).order_by(AccessLog.timestamp.desc()).limit(15).all()
    result = []
    for l in logs:
        if l.event_type in ('login', 'admin_login'):
            log_type = 'success'
            msg = f"Logged in from {l.location or l.ip_address}"
        elif l.event_type == 'logout':
            log_type = 'info'
            msg = "Logged out"
        elif l.event_type == 'file_access':
            log_type = 'info'
            msg = f"Accessed file: {l.detail}"
        elif l.event_type == 'behavior_anomaly':
            log_type = 'alert'
            msg = f"Behavior Anomaly detected: {l.detail}"
        elif l.event_type == 'behavior_verified':
            log_type = 'success'
            msg = f"Security verification challenge solved: {l.detail}"
        elif l.event_type == 'account_locked':
            log_type = 'alert'
            msg = f"Account locked: {l.detail}"
        else:
            log_type = 'info'
            msg = l.detail or l.event_type
            
        result.append({
            'timestamp': to_ist_filter(l.timestamp, '%H:%M:%S'),
            'message': msg,
            'type': log_type
        })
    return jsonify(result)


@app.route('/resource/api/userdata')
def resource_userdata():
    if not session.get('user'):
        return jsonify({'error': 'Not authenticated', 'redirect': url_for('login')}), 401
    return jsonify({'user': session['user'], 'trust_score': 0.82, 'data': 'sample protected payload'})


if __name__ == '__main__':
    # host='0.0.0.0' so it's reachable from other machines once this runs on
    # the college server, not just localhost. Keep debug=True only while
    # developing — turn it off before anyone outside your team can reach it.
    app.run(host='0.0.0.0', port=5000, debug=True)
