

squid-project/
│
├── app/
│   ├── main.py              # FastAPI REST API
│   └── streamlit_app.py     # Streamlit interface
│
├── data/
│   ├── raw/                 # Raw data (e.g., images)
│   └── processed/           # Preprocessed data
│
├── models/                  # Trained models
│
├── src/
│   ├── data_preprocessing.py  # Data preprocessing scripts
│   ├── train.py               # Model training
│   ├── evaluate.py            # Model evaluation
│   └── predict.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── dvc.yaml
└── README.md