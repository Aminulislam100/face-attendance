import io
import re
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from github_storage import (
    list_folder,
    load_dataframe,
    save_dataframe,
    upload_bytes,
    download_bytes,
)
from face_engine import prepare_face, face_to_jpeg, build_lbph_model, predict


st.set_page_config(
    page_title="Automated Face Attendance",
    page_icon="📷",
    layout="wide",
)

PEOPLE_PATH = "database/people.csv"
ATTENDANCE_PATH = "database/attendance.csv"

PEOPLE_COLUMNS = [
    "number", "name", "position", "folder", "created_at"
]

ATTENDANCE_COLUMNS = [
    "number", "name", "position", "date", "time", "status"
]


def clean_number(value):
    value = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(
            "Number/ID can contain only letters, numbers, _ and -"
        )
    return value


def load_people():
    return load_dataframe(PEOPLE_PATH, PEOPLE_COLUMNS)


def load_attendance():
    return load_dataframe(ATTENDANCE_PATH, ATTENDANCE_COLUMNS)


def show_github_error(e):
    st.error("GitHub update failed.")
    st.exception(e)


def load_all_face_samples(people_df):
    """
    Download all saved face images from GitHub and create LBPH training samples.
    """
    samples = []
    label_map = {}
    label_id = 0

    for _, person in people_df.iterrows():
        number = str(person["number"]).strip()
        folder = str(person["folder"]).strip()

        label_id += 1
        label_map[label_id] = person.to_dict()

        entries = list_folder(folder)

        for entry in entries:
            if entry.get("type") != "file":
                continue

            path = entry.get("path", "")
            if not path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            raw = download_bytes(path)
            if not raw:
                continue

            array = np.frombuffer(raw, dtype=np.uint8)
            gray_face = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)

            if gray_face is None:
                continue

            gray_face = cv2.equalizeHist(gray_face)
            gray_face = cv2.resize(gray_face, (200, 200))

            samples.append((label_id, gray_face))

    return samples, label_map


@st.cache_resource(show_spinner="Loading face database...")
def cached_model_and_map(people_signature, people_records):
    """
    people_signature is used to invalidate Streamlit's resource cache
    when the people list changes.
    """
    people_df = pd.DataFrame(people_records, columns=PEOPLE_COLUMNS)
    samples, label_map = load_all_face_samples(people_df)
    model = build_lbph_model(samples)
    return model, label_map, len(samples)


def model_from_current_database():
    people = load_people()

    if people.empty:
        return None, {}, 0

    signature = tuple(
        (
            str(row["number"]),
            str(row["folder"]),
            str(row["created_at"]),
        )
        for _, row in people.iterrows()
    )

    model, label_map, sample_count = cached_model_and_map(
        signature,
        people[PEOPLE_COLUMNS].astype(str).values.tolist(),
    )

    return model, label_map, sample_count


def mark_present(person):
    attendance = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    existing = attendance[
        (attendance["number"].astype(str) == str(person["number"])) &
        (attendance["date"].astype(str) == today)
    ]

    if not existing.empty:
        return False, existing.iloc[0].to_dict()

    new_row = pd.DataFrame([{
        "number": str(person["number"]),
        "name": str(person["name"]),
        "position": str(person["position"]),
        "date": today,
        "time": now_time,
        "status": "Present",
    }])

    updated = pd.concat([attendance, new_row], ignore_index=True)

    save_dataframe(
        ATTENDANCE_PATH,
        updated[ATTENDANCE_COLUMNS],
        f"Attendance: {person['name']} ({person['number']}) {today} {now_time}",
    )

    return True, new_row.iloc[0].to_dict()


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("📷 Attendance System")

page = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Add New Person",
        "Mark Attendance",
        "Attendance Sheet",
        "People Database",
    ],
)

threshold = st.sidebar.slider(
    "Face match threshold",
    min_value=40.0,
    max_value=110.0,
    value=75.0,
    step=1.0,
    help="Lower is stricter for LBPH. Start around 70–80 and test with your camera.",
)

# -----------------------------
# Dashboard
# -----------------------------

if page == "Dashboard":
    st.title("📊 Automated Face Attendance")

    try:
        people = load_people()
        attendance = load_attendance()

        c1, c2, c3 = st.columns(3)
        c1.metric("Registered People", len(people))

        today = datetime.now().strftime("%Y-%m-%d")
        today_records = attendance[
            attendance["date"].astype(str) == today
        ]

        c2.metric("Today's Present", len(today_records))
        c3.metric("Total Records", len(attendance))

        st.info(
            "GitHub is being used as the permanent storage layer. "
            "New people, face images and attendance records are committed "
            "to the repository."
        )

        st.subheader("Today's Attendance")
        if today_records.empty:
            st.write("No attendance yet today.")
        else:
            st.dataframe(
                today_records.sort_values("time"),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as e:
        show_github_error(e)


# -----------------------------
# Add New Person
# -----------------------------

elif page == "Add New Person":
    st.title("👤 Add New Person")

    st.caption(
        "Use one clear face only. Keep the camera in front of the face."
    )

    name = st.text_input("Name", placeholder="Aminul Islam")
    number = st.text_input("Number / ID", placeholder="001")
    position = st.text_input("Position", placeholder="Teacher")

    photo = st.camera_input("Take a face photo")

    if st.button("✅ Register Person", type="primary"):
        if not name.strip():
            st.warning("Enter a name.")
            st.stop()

        if not number.strip():
            st.warning("Enter a number/ID.")
            st.stop()

        if not position.strip():
            st.warning("Enter a position.")
            st.stop()

        if photo is None:
            st.warning("Take a face photo first.")
            st.stop()

        try:
            number = clean_number(number)

            people = load_people()

            if not people.empty and (
                people["number"].astype(str).str.lower() == number.lower()
            ).any():
                st.error("This Number / ID is already registered.")
                st.stop()

            image_bytes = photo.getvalue()
            array = np.frombuffer(image_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)

            if bgr is None:
                st.error("Could not read the camera image.")
                st.stop()

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            face, count = prepare_face(rgb)

            if count == 0:
                st.error("No face found. Take another photo.")
                st.stop()

            if count > 1:
                st.error(
                    "More than one face found. Only one person should be visible."
                )
                st.stop()

            folder = f"database_faces/{number}"
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            face_path = f"{folder}/face_1.jpg"

            # Save face image to GitHub first.
            upload_bytes(
                face_path,
                face_to_jpeg(face),
                f"Add face data for {name} ({number})",
            )

            new_person = pd.DataFrame([{
                "number": number,
                "name": name.strip(),
                "position": position.strip(),
                "folder": folder,
                "created_at": now,
            }])

            updated_people = pd.concat(
                [people, new_person],
                ignore_index=True
            )

            save_dataframe(
                PEOPLE_PATH,
                updated_people[PEOPLE_COLUMNS],
                f"Register person {name} ({number})",
            )

            st.success(
                f"✅ {name} was registered successfully."
            )
            st.info(
                "The face image and person information have been saved "
                "to your GitHub repository."
            )

            st.cache_resource.clear()

        except Exception as e:
            show_github_error(e)


# -----------------------------
# Mark Attendance
# -----------------------------

elif page == "Mark Attendance":
    st.title("📷 Mark Attendance")

    people = load_people()

    if people.empty:
        st.warning("No people are registered yet.")
        st.stop()

    camera = st.camera_input(
        "Look at the camera and take a photo"
    )

    if camera is not None:
        image_bytes = camera.getvalue()
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)

        if bgr is None:
            st.error("Could not read camera image.")
            st.stop()

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        face, count = prepare_face(rgb)

        if count == 0:
            st.error("❌ No face found.")
            st.stop()

        if count > 1:
            st.error(
                "❌ Multiple faces detected. "
                "Please keep only one person in the camera."
            )
            st.stop()

        with st.spinner("Matching face with GitHub database..."):
            try:
                model, label_map, sample_count = model_from_current_database()

                if model is None or not label_map:
                    st.error("Face database is empty.")
                    st.stop()

                label, confidence = predict(
                    model,
                    face,
                    threshold=threshold,
                )

                if label is None:
                    st.error(
                        f"❌ Face not recognized. Match score: {confidence:.1f}"
                    )
                    st.caption(
                        "Try better lighting or adjust the threshold in the sidebar."
                    )
                    st.stop()

                person = label_map[label]

                st.success("✅ Face recognized!")

                c1, c2, c3 = st.columns(3)
                c1.metric("Name", person["name"])
                c2.metric("Number", person["number"])
                c3.metric("Position", person["position"])

                now = datetime.now()

                st.write(
                    f"📅 Date: **{now.strftime('%d-%m-%Y')}**"
                )
                st.write(
                    f"⏰ Time: **{now.strftime('%I:%M:%S %p')}**"
                )
                st.write(
                    f"Face match score: **{confidence:.1f}**"
                )

                created, record = mark_present(person)

                if created:
                    st.success(
                        "✅ Attendance saved to GitHub."
                    )
                else:
                    st.info(
                        f"ℹ️ Already marked Present today at {record['time']}."
                    )

            except Exception as e:
                show_github_error(e)


# -----------------------------
# Attendance Sheet
# -----------------------------

elif page == "Attendance Sheet":
    st.title("📄 Attendance Sheet")

    try:
        attendance = load_attendance()

        if attendance.empty:
            st.info("No attendance records yet.")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            selected_date = st.date_input(
                "Date",
                value=datetime.now().date()
            )

        with col2:
            search = st.text_input(
                "Search Name / Number / Position"
            ).strip().lower()

        filtered = attendance[
            attendance["date"].astype(str)
            == selected_date.strftime("%Y-%m-%d")
        ].copy()

        if search:
            mask = (
                filtered["name"].str.lower().str.contains(search, na=False) |
                filtered["number"].str.lower().str.contains(search, na=False) |
                filtered["position"].str.lower().str.contains(search, na=False)
            )
            filtered = filtered[mask]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name=f"attendance_{selected_date.strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
        )

    except Exception as e:
        show_github_error(e)


# -----------------------------
# People Database
# -----------------------------

elif page == "People Database":
    st.title("👥 People Database")

    try:
        people = load_people()

        if people.empty:
            st.info("No people registered.")
            st.stop()

        st.dataframe(
            people[
                ["number", "name", "position", "created_at"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        show_github_error(e)
