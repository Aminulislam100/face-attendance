import base64
import io
import requests
import streamlit as st
import pandas as pd


API_URL = "https://api.github.com"


def _config():
    repo = st.secrets["GITHUB_REPO"].strip().rstrip("/")
    token = st.secrets["GITHUB_TOKEN"].strip()
    branch = st.secrets.get("GITHUB_BRANCH", "main").strip()

    if "/" not in repo:
        raise ValueError("GITHUB_REPO must look like: username/repository")

    return repo, token, branch


def _headers():
    _, token, _ = _config()
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_file(path: str):
    """Return (bytes, sha) for a file. Return (None, None) if it does not exist."""
    repo, _, branch = _config()
    url = f"{API_URL}/repos/{repo}/contents/{path}"

    response = requests.get(
        url,
        headers=_headers(),
        params={"ref": branch},
        timeout=30,
    )

    if response.status_code == 404:
        return None, None

    response.raise_for_status()
    data = response.json()

    if data.get("type") != "file":
        raise ValueError(f"{path} is not a file.")

    content = base64.b64decode(data["content"])
    return content, data["sha"]


def list_folder(path: str):
    """Return file entries in a GitHub repository directory."""
    repo, _, branch = _config()
    url = f"{API_URL}/repos/{repo}/contents/{path}"

    response = requests.get(
        url,
        headers=_headers(),
        params={"ref": branch},
        timeout=30,
    )

    if response.status_code == 404:
        return []

    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list):
        return []

    return data


def upload_or_update_file(path: str, content: bytes, message: str, sha=None):
    """Create/update one file in GitHub."""
    repo, _, branch = _config()
    url = f"{API_URL}/repos/{repo}/contents/{path}"

    payload = {
        "message": message,
        "content": base64.b64encode(content).decode("utf-8"),
        "branch": branch,
    }

    if sha:
        payload["sha"] = sha

    response = requests.put(
        url,
        headers=_headers(),
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def save_dataframe(path: str, df: pd.DataFrame, message: str):
    """Save a dataframe as CSV in GitHub."""
    old_content, sha = get_file(path)

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    upload_or_update_file(
        path=path,
        content=buffer.getvalue().encode("utf-8"),
        message=message,
        sha=sha,
    )


def load_dataframe(path: str, columns):
    """Load CSV from GitHub, returning an empty dataframe when absent."""
    content, _ = get_file(path)

    if content is None:
        return pd.DataFrame(columns=columns)

    text = content.decode("utf-8-sig")
    if not text.strip():
        return pd.DataFrame(columns=columns)

    return pd.read_csv(io.StringIO(text), dtype=str).fillna("")


def download_bytes(path: str):
    content, _ = get_file(path)
    return content


def upload_bytes(path: str, content: bytes, message: str):
    _, sha = get_file(path)
    return upload_or_update_file(
        path=path,
        content=content,
        message=message,
        sha=sha,
    )
