# YouTube Viral Predictor – Vercel deployment

## 1. Train and export the model (once, locally)
    pip install pandas scikit-learn
    python train_export.py path/to/USvideos.csv     # creates api/model.json

## 2. Test locally (optional)
    npm i -g vercel
    vercel dev

## 3. Deploy
    vercel login
    vercel            # preview deployment
    vercel --prod     # production deployment

Or push this folder to GitHub and import it at https://vercel.com/new (no build settings needed).

## API
POST /api/predict  with JSON
    {"likes":20000,"dislikes":500,"comment_count":3000,"title_length":45,"days_since_publish":2}
returns predicted_views, viral_probability, is_viral, viral_threshold_views.
GET /api/predict returns model metrics (health check).

---
## Streamlit front end (professional UI)
Streamlit needs a long-running server, so it can NOT run on Vercel. Host it on Streamlit Community Cloud (free).

Run locally from the project root (after train_export.py has created api/model.json):

    pip install -r streamlit_app/requirements.txt
    streamlit run streamlit_app/app.py

Deploy: push this whole folder (including api/model.json) to a GitHub repo, then at
https://share.streamlit.io choose "Create app", pick the repo, and set the main file path to
`streamlit_app/app.py`. The Vercel API keeps working independently.
