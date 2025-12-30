# 🍼 Baby Weight & Daily Intake Tracker (Streamlit)

A simple, local web app you run on your computer to track:
- Baby feeds (breast or bottle)
- Pumping amounts
- Diaper changes
- Baby weights (including “naked” vs “clean diaper”)
- Daily intake totals + an optional daily goal
- Notes/advice and thank-you reminders
- A groceries list (with priority items)

All data is saved **locally on your computer** in small `.json` files (no accounts, no cloud).

---
## What you need

- A computer (Mac / Windows / Linux)
- **Python 3.10+** installed  

---
## Step 1 — Download this project

### Option A: Download as ZIP (easiest)
1. On the GitHub page, click **Code → Download ZIP**
2. Unzip it somewhere you can find it (like Desktop)

### Option B: Clone with Git (optional)
If you already know Git:
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

---
## Step 2 — Open a terminal (command line)

- **Mac:** Applications → Utilities → **Terminal**
- **Windows:** Start Menu → **Command Prompt** or **PowerShell**
- **Linux:** Open **Terminal**

Then navigate to the project folder.

Example (Mac/Linux):
cd path/to/your-repo-folder

Example (Windows PowerShell):
cd path\to\your-repo-folder

---
## Step 3 — Create a virtual environment (recommended)

This keeps the app’s packages separate from everything else on your system.

### Mac / Linux
python3 -m venv .venv
source .venv/bin/activate

### Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

---
## Step 4 — Install the requirements

Run:
pip install -r requirements.txt

---
## Step 5 — Run the app
streamlit run app.py

A browser window should open automatically. If it doesn’t, the terminal will show a “Local URL” like:
- `http://localhost:8501`

Open that link in your browser.

To stop the app, go back to the terminal and press **Ctrl + C**.

---
## Where your data is stored

The app saves everything in a folder called:

data/

Inside it are files like:
- `events.json` (feeds, diapers, weights, pumping, vitamin D)
- `settings.json` (saved settings like goal and last tare)
- `thank_you.json`
- `advice.json`
- `groceries.json`

**Backing up is easy:** copy the entire `data/` folder somewhere safe.

---
## Basic usage

### Snapshot tab
Quick view of:
- last naked weights
- breastfeeding side reminder
- vitamin D “given today” button
- today’s intake + pace + goal progress
- priority groceries summary

### Calculator tab
Enter a baby weight (or pick recent naked weights) to see:
- unit conversions
- daily intake estimates
- set today’s feeding goal

### Event Tracker tab
Log and edit:
- diaper changes
- feeds (bottle or breast, optional weights)
- pumping
- random weights

### Graph tab
Charts for:
- weight over time (g or lb/oz)
- daily intake curves (last 3 days overlaid)
- optional goal line for today

### Thank You Notes / Notes & Advice / Groceries tabs
Lightweight lists you can add to, edit, and delete.

---
## Troubleshooting

### “streamlit: command not found”
Make sure your virtual environment is activated (`(.venv)` shows in terminal)

### “No module named …”
You likely didn’t install requirements.

### The app runs but no data appears
That’s normal on first run. Start adding events in **Event Tracker**.

### I want to reset everything
Close the app, then delete the `data/` folder (or just the JSON files inside it).

---
## Privacy
This app does **not** send your data anywhere. It runs on your computer and saves to local files only.