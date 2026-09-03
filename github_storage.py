import base64
import io

import pandas as pd
import requests
import streamlit as st


GITHUB_API = (
    "https://api.github.com"
)


# =========================================================
# CONFIG
# =========================================================

def get_config():

    if "GITHUB_REPO" not in st.secrets:

        raise RuntimeError(
            "GITHUB_REPO is missing from Streamlit Secrets."
        )

    if "GITHUB_TOKEN" not in st.secrets:

        raise RuntimeError(
            "GITHUB_TOKEN is missing from Streamlit Secrets."
        )

    repo = (
        st.secrets["GITHUB_REPO"]
        .strip()
        .rstrip("/")
    )

    token = (
        st.secrets["GITHUB_TOKEN"]
        .strip()
    )

    branch = (
        st.secrets
        .get("GITHUB_BRANCH", "main")
        .strip()
    )

    return (
        repo,
        token,
        branch
    )


# =========================================================
# HEADERS
# =========================================================

def get_headers():

    _, token, _ = get_config()

    return {

        "Accept":
            "application/vnd.github+json",

        "Authorization":
            f"Bearer {token}",

        "X-GitHub-Api-Version":
            "2022-11-28"
    }


# =========================================================
# GET FILE
# =========================================================

def get_file(path):

    repo, _, branch = (
        get_config()
    )

    url = (
        f"{GITHUB_API}/repos/"
        f"{repo}/contents/{path}"
    )

    response = requests.get(

        url,

        headers=get_headers(),

        params={
            "ref": branch
        },

        timeout=30
    )

    if response.status_code == 404:

        return None, None

    response.raise_for_status()

    data = response.json()

    if data.get("type") != "file":

        raise RuntimeError(
            f"GitHub path is not a file: {path}"
        )

    content = base64.b64decode(
        data["content"]
    )

    return (
        content,
        data["sha"]
    )


# =========================================================
# LIST FOLDER
# =========================================================

def list_folder(path):

    repo, _, branch = (
        get_config()
    )

    url = (
        f"{GITHUB_API}/repos/"
        f"{repo}/contents/{path}"
    )

    response = requests.get(

        url,

        headers=get_headers(),

        params={
            "ref": branch
        },

        timeout=30
    )

    if response.status_code == 404:

        return []

    response.raise_for_status()

    data = response.json()

    if not isinstance(
        data,
        list
    ):

        return []

    return data


# =========================================================
# CREATE OR UPDATE FILE
# =========================================================

def upload_or_update_file(
    path,
    content,
    message,
    sha=None
):

    repo, _, branch = (
        get_config()
    )

    url = (
        f"{GITHUB_API}/repos/"
        f"{repo}/contents/{path}"
    )

    payload = {

        "message":
            message,

        "content":
            base64.b64encode(
                content
            ).decode("utf-8"),

        "branch":
            branch
    }

    if sha:

        payload["sha"] = sha

    response = requests.put(

        url,

        headers=get_headers(),

        json=payload,

        timeout=60
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# SAVE DATAFRAME
# =========================================================

def save_dataframe(
    path,
    dataframe,
    message
):

    _, sha = get_file(
        path
    )

    csv_buffer = (
        dataframe.to_csv(
            index=False
        )
    )

    upload_or_update_file(

        path=path,

        content=csv_buffer.encode(
            "utf-8"
        ),

        message=message,

        sha=sha
    )


# =========================================================
# LOAD DATAFRAME
# =========================================================

def load_dataframe(
    path,
    columns
):

    content, _ = (
        get_file(path)
    )

    if content is None:

        return pd.DataFrame(
            columns=columns
        )

    text = (
        content
        .decode("utf-8-sig")
    )

    if not text.strip():

        return pd.DataFrame(
            columns=columns
        )

    return pd.read_csv(
        io.StringIO(text),
        dtype=str
    ).fillna("")


# =========================================================
# UPLOAD BYTES
# =========================================================

def upload_bytes(
    path,
    content,
    message
):

    _, sha = get_file(
        path
    )

    return upload_or_update_file(

        path=path,

        content=content,

        message=message,

        sha=sha
    )


# =========================================================
# DOWNLOAD BYTES
# =========================================================

def download_bytes(path):

    content, _ = (
        get_file(path)
    )

    return content
