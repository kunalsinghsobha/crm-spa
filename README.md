# SPA Cycle Time Dashboard (Streamlit)

A lightweight, real‑time **app** for tracking SPA cycle times and delays across stages (S1–S4) using your daily Excel report.

**Why this app?**
- Upload the Excel → the app applies SLA rules and flags delays instantly.
- Toggle **Executed vs Pending** cases.
- Filter by **Project/Tower/Unit/Booking**, **Stage**, **Severity**, and **Date**.
- Drill down into a single booking with a timeline and reasons.
- One‑click CSV exports for circulation.

---

## ✨ Features
- **Topline KPIs** for delayed stages and missing data
- **Funnel view** with business-friendly labels
- **SLA Compliance Snapshot** with legend (S1–S4)
- **Executed ↔ Pending** switch
- Pending classification with **pending days** and **severity bands**
- Robust parsing of mixed date formats; safe handling of missing values
- Instant filtering and fast refresh (Streamlit caching)

---

## 🧠 SLA Logic (business rules encoded)
- **S1 – Eligibility**: `SPA Eligibility - SalesOps Approval ≤ 14 days` (else delayed)
- **S2 – SPA Sending**: `SPA Sent - max(Eligibility, Floorplan Upload, Registration) ≤ 3 days` (else delayed)
- **S3 – Customer Return (CRM)**: `CRM Assurance - SPA Sent ≤ 14 days` (15–60 delayed; **>60 critical**)
- **S4 – Execution**: `SPA Executed - CRM Assurance ≤ 3 days` (else delayed)

Missing dates are surfaced as **“Oopsie! Data not present”** to avoid false conclusions.

---

## 🗂️ Expected Columns (exact or similar names)
- Project, Tower Name, Unit Name, Booking Name, RERA Number
- SalesOps Assurance Approval Date
- SPA Eligibility Date
- Floor Plan Upload Date
- Registration date
- SPA Sent Date
- SPA Sent to CRM OPS Assurance Date
- SPA Executed Date
- Date of Pre-Registration Initiation

> Column names can vary slightly; the app matches them flexibly.

---

## 🛠️ Local Development

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run
```bash
streamlit run app.py
```
Then open the local URL that Streamlit prints (e.g., http://localhost:8501).

---

## ☁️ Deploy on Streamlit Cloud (via GitHub)

1. **Create a GitHub repo** (public or private).  
   Add these files:
   - `app.py` (your Streamlit application file)
   - `requirements.txt` (see below)
   - `README.md` (this file)
   - Optional: `.streamlit/config.toml` for theming

2. **Push to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit: SPA Cycle Time Dashboard"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

3. **Deploy on Streamlit Cloud**  
   - Go to https://share.streamlit.io  
   - Click **New app** → pick your repo/branch → set **Main file path** = `app.py` → **Deploy**  
   - The app will build automatically and give you a public URL

4. **Use the app**
   - Click **Browse files** inside the app and upload your **Excel (.xlsx)** report.
   - Apply filters / toggle **Executed ↔ Pending** as needed.

> No Procfile required. Streamlit Cloud auto-detects `requirements.txt` and `app.py`.

---

## 🧾 Requirements

See `requirements.txt` in this repo for exact packages. Pinning to broadly compatible, modern versions.

---

## 🧹 Project Structure
```
spa_cycle_time_dashboard_repo/
├─ app.py                    # Your Streamlit app (paste the final code here)
├─ requirements.txt          # Python dependencies
├─ README.md                 # This file
├─ .streamlit/
│  └─ config.toml            # (Optional) Theme + server tweaks
└─ .gitignore                # (Optional) keep venv & caches out of git
```

---

## 🎨 Optional: Theme (.streamlit/config.toml)
This repo includes a sample theme. Tweak colors as you like.

---

## 🧩 Troubleshooting
- **App can’t find columns**: Check your Excel headers; small name differences are OK, but the fields should exist.
- **Dates look wrong**: The app supports day-first formats and Excel serials; ensure your sheet isn’t mixing text like “N/A” with dates in the same column.
- **Slow first load**: Streamlit Cloud needs a few seconds to cold-start; subsequent interactions are cached.
- **Large files**: Try removing unused columns or splitting the report—Streamlit is memory-friendly but not a data warehouse.

---

## 🔐 Security Notes
- This MVP loads data interactively from user uploads; **no data is stored** on the server between sessions by design.
- Do not commit real customer data to the repo.

---

## 📣 Support
If you need help adjusting labels or adding new charts later, open an issue in your GitHub repo.
