# Scooter Wheels Scheduler (Streamlit)

Interactive Gantt of a factory scheduling puzzle (scooter wheels), with a prompt bar for future command-based edits.

## Local dev

```bash
# clone
git clone https://github.com/<your-username>/scooter-scheduler.git
cd scooter-scheduler

# (optional) create data if not present
python scripts/generate_sample_data.py

# install & run
pip install -r requirements.txt
streamlit run app.py
