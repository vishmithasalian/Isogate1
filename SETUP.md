# NeuroGate — setup notes

## Install
```
pip install Flask Flask-SQLAlchemy Flask-Limiter scikit-learn numpy
```

## Run
Delete the old `neurogate.db` if it's from before the activity-log/behavior
update (new tables were added), then:
```
python app.py
```

## Accounts and admin access
Admin access is now completely separate from regular login — it lives at
`/admin/login`, with its own page, its own session, and its own lockout/OTP
flow. A regular account signing in at `/login` never sees admin pages, even
if that same email is in the admin list.

Who counts as admin is controlled by `NEUROGATE_ADMIN_EMAILS` (comma
separated). By default that's:
```
tanvisuares2005@gmail.com,vishmithasalian@gmail.com
```
Add more later by setting the env var, e.g.:
```
export NEUROGATE_ADMIN_EMAILS="tanvisuares2005@gmail.com,vishmithasalian@gmail.com,someone.else@gmail.com"
```
On startup the app auto-creates an account for any admin email that doesn't
exist yet (password `ChangeMe123!` for newly added ones — change it via
"Forgot password" on the admin login page the first time you sign in).

Seeded demo accounts on first run:
- `norman@gmail.com` / `nor123` (regular user)
- `tanvisuares2005@gmail.com` / `admin123` (admin)
- `vishmithasalian@gmail.com` / `admin123` (admin)

## Turning on real email (OTP + admin lockout alerts)
Both the "forgot password" OTP and the 4-failed-attempts admin alert now go
through the *server*, not EmailJS, so they can't be spoofed or skipped from
the browser. To actually send mail, set these environment variables before
running the app:

```
# macOS/Linux
export NEUROGATE_SMTP_USER="yourbot@gmail.com"
export NEUROGATE_SMTP_PASS="16-char app password"
export NEUROGATE_ADMIN_EMAIL="tanvisuares2005@gmail.com"

# Windows PowerShell
$env:NEUROGATE_SMTP_USER="yourbot@gmail.com"
$env:NEUROGATE_SMTP_PASS="16-char app password"
$env:NEUROGATE_ADMIN_EMAIL="tanvisuares2005@gmail.com"
```

To get a Gmail App Password: turn on 2-Step Verification on the sending
account, then visit https://myaccount.google.com/apppasswords and generate
one for "Mail". Use that 16-character password, not your normal login
password.

If you don't set these, the app still runs — OTP codes and alert emails just
get written to the server log instead of actually being mailed, so you can
test the flow end-to-end before wiring up real SMTP.

## What changed vs. your old code
- Passwords are now hashed (`werkzeug.security`) instead of stored in plain
  text — added a `password_hash` column.
- `/store-otp` was renamed `/send-otp` and now **generates** the OTP
  server-side and emails it, instead of trusting a code the client made up
  and posted to you (that was a bug — anyone could set `user_otp` to
  whatever they wanted and it would validate).
- OTPs expire after 5 minutes.
- After 4 failed logins the account is locked (`locked=True`) and an alert
  email fires to `NEUROGATE_ADMIN_EMAIL`. An admin can unlock it from
  `/admin`, or the user can unlock it themselves via password reset.
- Login errors are now the same generic message whether the email exists or
  not, so the form can't be used to check which emails have accounts.
- Added `/admin` (accounts + lockout status + unlock button) and a shared
  `neurogate-theme` dark/light toggle across `index.html`, `login.html`,
  `dashboard.html`, and `admin.html`.
- Admin login is now fully separate (`/admin/login`, its own session key,
  its own template) instead of being gated by an `is_admin` flag on the
  same session as a normal user login.
- Added a lightweight behavioral-anomaly check: the dashboard already
  tracked typing rhythm and mouse speed, now it periodically reports that
  to `/api/behavior`, which compares it to a rolling per-user baseline. A
  big deviation gets logged as `behavior_anomaly` and emails the admin —
  useful for catching a hijacked session where someone else is at the
  keyboard. This is a simple statistical check, not the trained model in
  `neurogate_trust_model.h5` — say the word if you want help wiring that
  model in instead once you confirm its exact input feature order.
- Added precise-location capture: the dashboard asks the browser for GPS
  coordinates (standard permission prompt) and reverse-geocodes them via
  OpenStreetMap, shown in `/admin/activity` alongside the coarser
  IP-based location that was already there. If someone denies the
  permission prompt, the IP-based location still gets recorded.
- `app.run()` now binds to `0.0.0.0` so it'll be reachable once this moves
  to the college server, not just localhost. Before that move: turn
  `debug=True` off, and switch `SQLALCHEMY_DATABASE_URI` off SQLite to
  something that handles concurrent users better if many people will hit
  it at once (Postgres/MySQL) — SQLite is fine for testing but not built
  for that.

## Fixes in this round
- `/` was redirecting to a landing page instead of straight to login — reverted,
  it goes to `/login` again for signed-out visitors.
- Clicking Dashboard/Files from the admin panel used to bounce you back to
  login, because admin sessions and regular-user sessions were checked
  separately. Both routes now accept either session, so an admin can browse
  the org platform too.
- `/admin` now shows a "Last File Accessed" column per user instead of
  making you dig through the full activity log.
- Added rate limiting (`Flask-Limiter`) on `/login`, `/admin/login`, and
  `/send-otp` — 10/min, 10/min, 5/min per IP — so the failed-attempt lockout
  isn't the only thing standing between an attacker and rapid-fire guessing.

## About the feature list you sent
That's a great roadmap, and a few of the items are already done rather than
still on the list:
- Password hashing — already `werkzeug.security`, not plaintext.
- OTP expiration — already 5 minutes, checked server-side.
- A basic anomaly signal — already there (`/api/behavior`, `/admin/behavior`),
  just simpler than the Isolation Forest one you described.

The rest is real, substantial work — each of these is its own multi-hour
build, not a quick add-on:
- **Adaptive MFA on high behavioral risk even with a correct password** —
  natural next step, reuses infrastructure that already exists.
- **Isolation Forest / One-Class SVM anomaly model** — needs a labeled
  training set of "normal" logins per user before it's better than the
  current statistical check; worth doing once you have real usage data.
- **Impossible-travel detection** — doable now that location is logged;
  needs a distance/time-between-logins calculation.
- **Chart.js live biometric graphs, "Signature Matching..." UI** — pure
  frontend polish, good demo value, moderate effort.
- **WebAuthn/Passkeys, JWT sessions** — bigger architectural changes, would
  touch most of the auth flow.

Tell me which one to build next and I'll do that one properly rather than
all of them shallowly.

## Real ML anomaly model, Chart.js, and the location map
- **Isolation Forest**: `/api/behavior` now scores each sample with a fresh
  `sklearn.ensemble.IsolationForest` trained on that user's own last 60
  behavioral readings (typing flight time + mouse velocity). Below 8 stored
  samples it falls back to the old percentage-deviation check so brand new
  accounts aren't flagged for having "no history" — the badge/column on
  `/admin/behavior` shows which method produced each reading (`learning`,
  `statistical`, or `isolation_forest`).
- Every raw reading is now kept in a new `BehaviorSample` table (not just a
  rolling average), which is what makes the charts possible.
- **Chart.js**: your own dashboard now has a live "Behavioral Risk Score"
  line chart. `/admin/behavior` has the same chart with a dropdown to pick
  any user and see their risk trend over time, plus the existing live table.
- **Location map**: `/admin/map` — a real map (Leaflet + OpenStreetMap
  tiles, free, no API key) with a marker per recent login. Cyan markers are
  precise (GPS, permission granted); grey markers are the coarser IP-based
  fallback. Click a marker for the user, place, and time.

None of this needs external API keys — Nominatim, ip-api.com, and the
OpenStreetMap tiles are all free/keyless, which is why there's rate-limit
etiquette baked into the code (custom User-Agent, short timeouts, no
hammering their endpoints).
