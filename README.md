# Automated Face Attendance — GitHub + Streamlit

This project stores people data, face images, and attendance CSV files in the GitHub repository through the GitHub REST API.

## Repository layout

```text
face-attendance/
├── app.py
├── face_engine.py
├── github_storage.py
├── requirements.txt
├── .gitignore
├── README.md
├── database/
│   ├── people.csv
│   └── attendance.csv
├── database_faces/
└── .streamlit/
    └── config.toml
```

## Streamlit Secrets

Add these in Streamlit Cloud -> App -> Settings -> Secrets:

```toml
GITHUB_TOKEN = "YOUR_FINE_GRAINED_TOKEN"
GITHUB_REPO = "YOUR_USERNAME/YOUR_REPOSITORY"
GITHUB_BRANCH = "main"
```

Do NOT commit `.streamlit/secrets.toml`.

## Features

- Add a person with name, number, position and face photo
- Saves face data to `database_faces/<number>/`
- Saves people information to `database/people.csv`
- Recognizes a single face from camera input
- Adds date/time attendance automatically
- Prevents duplicate attendance on the same date
- Shows/filter attendance and downloads CSV

## Important

This is a small-scale prototype. GitHub is not a high-concurrency database. For many simultaneous users or large face datasets, use a real database/object-storage architecture.

Face data can be biometric information. Keep the repository private, restrict access, and only collect/use face data with the required consent and legal basis.
