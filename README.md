# ChurnGuardPro - Customer Churn Prediction Dashboard

ChurnGuardPro is an end-to-end Machine Learning engineering project that transforms exploratory customer analysis into a production-ready, interactive web dashboard. It features robust pipelines, MLflow experiment tracking, and a glassmorphic dark-theme executive user interface.

---

## 📂 Project Structure

```
├── data/
│   └── Telco-Customer-Churn.csv       # Raw dataset
├── artifacts/
│   ├── train.csv                      # Partitioned training data
│   └── test.csv                       # Partitioned testing data
├── models/
│   ├── churn_model.pkl                # Trained Gradient Boosting weights
│   ├── scaler.pkl                     # Fitted StandardScaler instance
│   ├── model_columns.pkl              # Schema categories for dummy alignment
│   ├── metrics.json                   # Current pipeline evaluation metrics
│   └── feature_importance.json        # Top 5 predictive churn drivers
├── src/
│   └── customer_churn_prediction/     # Pipeline package
│       ├── components/                # Ingestion, transformation, training
│       ├── pipelines/                 # Training and prediction pipelines
│       ├── logger.py                  # Structured logger
│       ├── exception.py               # Custom Exception Handler
│       └── utils.py                   # Serialization utilities
├── templates/
│   └── index.html                     # Bento dashboard HTML UI
├── static/
│   ├── style.css                      # Glassmorphic dark styling rules
│   └── script.js                      # Chart.js and SSE log streaming logic
├── customer-churn.ipynb               # Original Notebook (preserved for EDA)
├── app.py                             # Flask server routes
├── main.py                            # CLI training execution script
├── Dockerfile                         # Container configuration
├── requirements.txt                   # Dependency list
└── setup.py                           # Packaging setup
```

---

## ⚡ How to Run the Project

### Prerequisites
Make sure Python 3.9+ is installed on your local system.

### 1. Set Up the Virtual Environment
Open a terminal in the project root directory and run:
```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment (Windows)
.venv\Scripts\activate

# Activate the environment (macOS/Linux)
source .venv/bin/activate
```

### 2. Install Dependencies
Install all the required python packages and register the local package namespaces:
```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Run the Training Pipeline (Optional)
The pre-trained model weights are already provided in the `models/` directory. However, to re-run the end-to-end pipeline (Data Ingestion ➔ Data Transformation ➔ Model Training with SMOTE ➔ MLflow Tracking), run:
```bash
python main.py
```
This logs the metrics and parameters in a local SQLite database (`mlflow.db`) under the experiment `Telco_Customer_Churn_Prediction`.

### 4. Start the Flask Application
Run the Flask server backend locally:
```bash
python app.py
```
By default, the server starts up on port **`5001`**. Once running, navigate to:
👉 **[http://127.0.0.1:5001](http://127.0.0.1:5001)**

### 5. Running with Docker (Containerized Mode)
To run the entire application inside a container (which handles compilation of all dependencies automatically):
```bash
# Build the Docker image
docker build -t churnguard-pro .

# Run the container
docker run -p 5001:5001 churnguard-pro
```

---

## ⚠️ Running This Project on Another PC (Required Changes)

If you move/copy this repository to another machine, do these steps before running. This prevents absolute-path issues (for example `C:\Users\old-user\...`) and MLflow permission errors.

### 1. Remove machine-specific folders/files
Delete these if they exist in the copied project:

```bash
# Windows PowerShell
Remove-Item -Recurse -Force env, .venv, customer_churn.egg-info, customer_churn_prediction.egg-info
```

Notes:
- `env/` or `.venv/` contains interpreter paths from the old machine.
- `*.egg-info` can keep stale editable-install metadata.

### 2. Create and activate a fresh virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3. Reinstall project dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Reset MLflow local state (recommended on first run)

```bash
# Optional but recommended when repo was copied from another PC
Remove-Item -Recurse -Force mlruns -ErrorAction SilentlyContinue
Remove-Item -Force mlflow.db -ErrorAction SilentlyContinue
```

Why:
- Old MLflow experiments can point to non-writable artifact locations from another user profile.
- The training code now auto-falls back to a writable local experiment, but resetting gives a clean history.

### 5. Run training and app

```bash
python main.py
python app.py
```

### 6. If you still get permission errors
Check that the project is not under a protected directory and run PowerShell as a normal user from your project folder (for example under `D:\Projects\...`), then retry steps 2-5.

