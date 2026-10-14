# LoanLens  Capstone-05
Next.js + FastAPI + default/CV-tuned KNN. Educational only, not a lending system.

## Local backend (terminal 1)
Use Python 3.11 or 3.12. From this project folder:
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python -m uvicorn main:app --reload --port 8000
```
If activation is blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
API docs: http://localhost:8000/docs . Routes: GET /health, GET /metrics, POST /predict. The backend trains/caches both training-only models at startup; saved models are optional. No API key needed.

## Frontend (terminal 2)
Use Node.js 22 LTS. From this project folder:
```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```
Open http://localhost:3000. Model selector offers default and CV-tuned models. Unknown fields are imputed. Errors and loading states are included.

## Method/results
614 labeled rows. Loan_ID excluded. Target N=0, Y=1. Median numerical imputation, mode categorical imputation, one-hot encoding, StandardScaler applied to all transformed columns including dummies. This weighting affects distances. All preprocessing happens inside the CV pipeline. Stratified 80/20 split, random_state=42: 491 train, 123 test. Default KNeighborsClassifier() unchanged. Five-fold training-only CV compares k=3,5,7 and tunes over 3,5,7,9,11,15,21. Test does not select k. See artifacts/ for executed results. Unlabeled external test file is not used for accuracy.
Default accuracy: 78.86%. CV-selected k=15 test accuracy:73.17%. Tuning did NOT improve test accuracy. Tuned model finds only 5 of 38 rejection labels; accuracy alone is inadequate. Both matrices and k experiment chart are saved in the executed notebook.

## Notebook
backend/Loan_KNN_Analysis.ipynb is executed with outputs and visualizations. Run from backend or project root. In Colab, upload train.py first, then data/train.csv under filename train.csv when prompted.

## Deployment
Frontend project root: frontend. Set NEXT_PUBLIC_API_URL to HTTPS backend base URL before building. Backend root: backend. Install requirements.txt and include data/train.csv. Start with uvicorn main:app --host 0.0.0.0 --port $PORT . Set ALLOWED_ORIGINS to exact frontend origin (no trailing slash), comma-separated for multiple origins. Rebuild frontend after URL changes. Do not use localhost in deployed environment.

## Status/limitations
Training, CV, test evaluation, notebook plots, Python syntax, FastAPI endpoint smoke checks, and the Next.js production build have been validated. Browser rendering, full browser-to-backend integration, and cloud deployment have not been tested here. No live deployment or LinkedIn post created. No authentication, rate limiting, calibration, security review, fairness audit or lending validation. Do not enter real financial/identity data. Protected/sensitive fields included only to follow assignment; not an endorsement of their use in lending. Neighbor fractions are not calibrated probabilities. Dataset units are not verified. Source https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset has unknown license: confirm public redistribution permission.

## Submission
Test locally, deploy if desired, record form and prediction plus model results. Post screenshots/video on LinkedIn and submit that post URL. Use included caption template after replacing links and confirming functionality.
