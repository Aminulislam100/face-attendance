```python
import re
from datetime import datetime
from zoneinfo import ZoneInfo

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

from face_engine import (
    prepare_face,
    face_to_jpeg,
    build_lbph_model,
    predict,
    get_match_quality,
)


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Attendance System",
    page_icon="📷",
    layout="wide"
)


# =========================================================
# CONSTANTS
# =========================================================

PEOPLE_PATH = (
    "database/people.csv"
)

ATTENDANCE_PATH = (
    "database/attendance.csv"
)

PEOPLE_COLUMNS = [
    "number",
    "name",
    "position",
    "folder",
    "created_at"
]

ATTENDANCE_COLUMNS = [
    "number",
    "name",
    "position",
    "date",
    "time",
    "status"
]

# Bangladesh timezone
BANGLADESH_TZ = ZoneInfo(
    "Asia/Dhaka"
)


# =========================================================
# BANGLADESH TIME
# =========================================================

def get_now():

    return datetime.now(
        BANGLADESH_TZ
    )


# =========================================================
# FUNCTIONS
# =========================================================

def clean_number(value):

    value = str(value).strip()

    if not re.fullmatch(
        r"[A-Za-z0-9_-]+",
        value
    ):

        raise ValueError(
            "Number / ID can contain only "
            "letters, numbers, _ and -."
        )

    return value


def load_people():

    return load_dataframe(
        PEOPLE_PATH,
        PEOPLE_COLUMNS
    )


def load_attendance():

    return load_dataframe(
        ATTENDANCE_PATH,
        ATTENDANCE_COLUMNS
    )


def image_from_camera(
    camera_image
):

    image_bytes = (
        camera_image
        .getvalue()
    )

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    bgr = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if bgr is None:

        raise RuntimeError(
            "Could not read camera image."
        )

    return cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )


def get_face_samples():

    people = load_people()

    samples = []

    label_map = {}

    current_label = 0

    for _, person in people.iterrows():

        current_label += 1

        number = str(
            person["number"]
        ).strip()

        folder = str(
            person["folder"]
        ).strip()

        label_map[
            current_label
        ] = person.to_dict()

        files = list_folder(
            folder
        )

        for file_info in files:

            if file_info.get(
                "type"
            ) != "file":

                continue

            path = file_info.get(
                "path",
                ""
            )

            if not path.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            ):

                continue

            raw = download_bytes(
                path
            )

            if not raw:

                continue

            array = np.frombuffer(
                raw,
                dtype=np.uint8
            )

            face = cv2.imdecode(
                array,
                cv2.IMREAD_GRAYSCALE
            )

            if face is None:

                continue

            face = cv2.resize(
                face,
                (200, 200),
                interpolation=cv2.INTER_AREA
            )

            face = cv2.equalizeHist(
                face
            )

            samples.append(
                (
                    current_label,
                    face
                )
            )

    return (
        samples,
        label_map
    )


def train_model():

    samples, label_map = (
        get_face_samples()
    )

    if not samples:

        return (
            None,
            {},
            0
        )

    model = build_lbph_model(
        samples
    )

    return (
        model,
        label_map,
        len(samples)
    )


# =========================================================
# SAVE ATTENDANCE
# =========================================================

def save_attendance(person):

    attendance = load_attendance()

    # Bangladesh local time
    now = get_now()

    today = now.strftime(
        "%Y-%m-%d"
    )

    current_time = now.strftime(
        "%H:%M:%S"
    )

    # -----------------------------------------------------
    # IMPORTANT
    #
    # We intentionally DO NOT check whether this person
    # already has attendance today.
    #
    # Therefore the same person can scan multiple times
    # on the same day.
    # -----------------------------------------------------

    new_record = pd.DataFrame(
        [{
            "number":
                person["number"],

            "name":
                person["name"],

            "position":
                person["position"],

            "date":
                today,

            "time":
                current_time,

            "status":
                "Present"
        }]
    )

    updated = pd.concat(
        [
            attendance,
            new_record
        ],
        ignore_index=True
    )

    save_dataframe(

        ATTENDANCE_PATH,

        updated[
            ATTENDANCE_COLUMNS
        ],

        (
            "Attendance - "
            f"{person['name']} "
            f"{today} "
            f"{current_time}"
        )
    )

    return (
        True,
        new_record.iloc[0].to_dict()
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "📷 Attendance System"
)

menu = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Add New Person",
        "Mark Attendance",
        "Attendance Sheet",
        "People Database"
    ]
)

threshold = st.sidebar.slider(

    "Face distance threshold",

    min_value=40.0,

    max_value=100.0,

    value=75.0,

    step=1.0,

    help=(
        "Lower values are stricter. "
        "LBPH distance is not a percentage. "
        "Start around 70–75 and test."
    )
)


# =========================================================
# DASHBOARD
# =========================================================

if menu == "Dashboard":

    st.title(
        "📷 Automated Attendance System"
    )

    try:

        people = load_people()

        attendance = load_attendance()

        # Bangladesh date
        today = get_now().strftime(
            "%Y-%m-%d"
        )

        today_attendance = attendance[
            attendance["date"]
            .astype(str)
            == today
        ]

        # -------------------------------------------------
        # IMPORTANT
        #
        # One person can have multiple attendance records
        # in one day.
        #
        # So Today's Present counts UNIQUE people.
        # -------------------------------------------------

        if today_attendance.empty:

            today_present_people = 0

        else:

            today_present_people = (
                today_attendance["number"]
                .astype(str)
                .nunique()
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Registered People",
            len(people)
        )

        c2.metric(
            "Today's Present",
            today_present_people
        )

        c3.metric(
            "Total Attendance",
            len(attendance)
        )

        st.subheader(
            "Today's Attendance"
        )

        if today_attendance.empty:

            st.info(
                "No attendance today."
            )

        else:

            st.dataframe(

                today_attendance[
                    [
                        "number",
                        "name",
                        "position",
                        "date",
                        "time",
                        "status"
                    ]
                ],

                use_container_width=True,

                hide_index=True
            )

    except Exception as e:

        st.error(
            str(e)
        )


# =========================================================
# ADD NEW PERSON
# =========================================================

elif menu == "Add New Person":

    st.title(
        "👤 Add New Person"
    )

    st.info(
        "Register 3 face samples for better recognition. "
        "Take photos with slightly different angles or lighting."
    )

    name = st.text_input(
        "Name",
        placeholder="EVA"
    )

    number = st.text_input(
        "Number / ID",
        placeholder="002"
    )

    position = st.text_input(
        "Position",
        placeholder="Manager"
    )

    st.subheader(
        "Face Samples"
    )

    photo1 = st.camera_input(
        "Face Sample 1",
        key="face_sample_1"
    )

    photo2 = st.camera_input(
        "Face Sample 2",
        key="face_sample_2"
    )

    photo3 = st.camera_input(
        "Face Sample 3",
        key="face_sample_3"
    )

    if st.button(
        "✅ Register Person",
        type="primary"
    ):

        if not name.strip():

            st.warning(
                "Please enter the name."
            )

            st.stop()

        if not number.strip():

            st.warning(
                "Please enter Number / ID."
            )

            st.stop()

        if not position.strip():

            st.warning(
                "Please enter position."
            )

            st.stop()

        if not photo1:

            st.warning(
                "Face Sample 1 is required."
            )

            st.stop()

        try:

            number = clean_number(
                number
            )

            people = load_people()

            if not people.empty:

                duplicate = (
                    people["number"]
                    .astype(str)
                    .str.lower()
                    == number.lower()
                )

                if duplicate.any():

                    st.error(
                        "This Number / ID is already registered."
                    )

                    st.stop()

            photos = [
                photo1,
                photo2,
                photo3
            ]

            valid_faces = []

            for index, photo in enumerate(
                photos,
                start=1
            ):

                if photo is None:

                    continue

                image = image_from_camera(
                    photo
                )

                face, count = (
                    prepare_face(image)
                )

                if count == 0:

                    st.error(
                        f"Face Sample {index}: "
                        "No face found."
                    )

                    st.stop()

                if count > 1:

                    st.error(
                        f"Face Sample {index}: "
                        "More than one face found."
                    )

                    st.stop()

                valid_faces.append(
                    face
                )

            if len(valid_faces) == 0:

                st.error(
                    "No valid face samples found."
                )

                st.stop()

            folder = (
                f"database_faces/{number}"
            )

            # Bangladesh time
            now = get_now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with st.spinner(
                "Saving face samples to GitHub..."
            ):

                for index, face in enumerate(
                    valid_faces,
                    start=1
                ):

                    upload_bytes(

                        f"{folder}/face_{index}.jpg",

                        face_to_jpeg(face),

                        (
                            f"Add face sample {index} "
                            f"for {name}"
                        )
                    )

                new_person = pd.DataFrame(
                    [{
                        "number":
                            number,

                        "name":
                            name.strip(),

                        "position":
                            position.strip(),

                        "folder":
                            folder,

                        "created_at":
                            now
                    }]
                )

                updated_people = (
                    pd.concat(
                        [
                            people,
                            new_person
                        ],
                        ignore_index=True
                    )
                )

                save_dataframe(

                    PEOPLE_PATH,

                    updated_people[
                        PEOPLE_COLUMNS
                    ],

                    (
                        "Register person "
                        f"{name} ({number})"
                    )
                )

            st.success(
                f"✅ {name} registered successfully."
            )

            st.write(
                f"Face samples saved: "
                f"**{len(valid_faces)}**"
            )

            st.write(
                f"GitHub folder: "
                f"`{folder}`"
            )

        except Exception as e:

            st.error(
                "❌ Registration failed."
            )

            st.exception(e)


# =========================================================
# MARK ATTENDANCE
# =========================================================

elif menu == "Mark Attendance":

    st.title(
        "📷 Mark Attendance"
    )

    people = load_people()

    if people.empty:

        st.warning(
            "No registered people found."
        )

        st.stop()

    camera = st.camera_input(
        "Look at the camera and take a photo"
    )

    if camera is not None:

        try:

            image = image_from_camera(
                camera
            )

            face, count = (
                prepare_face(image)
            )

            if count == 0:

                st.error(
                    "❌ No face found."
                )

                st.stop()

            if count > 1:

                st.error(
                    "❌ Multiple faces detected. "
                    "Only one person should be visible."
                )

                st.stop()

            with st.spinner(
                "Matching face..."
            ):

                model, label_map, sample_count = (
                    train_model()
                )

            if model is None:

                st.error(
                    "Face database is empty."
                )

                st.stop()

            label, distance = predict(
                model,
                face,
                threshold
            )

            # -----------------------------------------
            # NOT RECOGNIZED
            # -----------------------------------------

            if label is None:

                st.error(
                    "❌ Face not recognized."
                )

                if distance is not None:

                    st.write(
                        f"Face distance: "
                        f"**{distance:.1f}**"
                    )

                    st.caption(
                        "Lower distance generally means "
                        "a better match."
                    )

                st.stop()

            person = label_map[
                label
            ]

            quality = (
                get_match_quality(
                    distance
                )
            )

            # Bangladesh current time
            now = get_now()

            # -----------------------------------------
            # RESULT
            # -----------------------------------------

            st.success(
                "✅ Face recognized!"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Name",
                person["name"]
            )

            c2.metric(
                "Number",
                person["number"]
            )

            c3.metric(
                "Position",
                person["position"]
            )

            st.write(
                f"📅 Date: "
                f"**{now.strftime('%d-%m-%Y')}**"
            )

            st.write(
                f"⏰ Time: "
                f"**{now.strftime('%I:%M:%S %p')}**"
            )

            st.write(
                f"🎯 Face distance: "
                f"**{distance:.1f}**"
            )

            st.write(
                f"⭐ Match quality: "
                f"**{quality}**"
            )

            st.caption(
                "Note: Face distance is not a percentage. "
                "Lower distance generally means a better match."
            )

            # -----------------------------------------
            # SAVE ATTENDANCE
            # -----------------------------------------

            saved, record = (
                save_attendance(
                    person
                )
            )

            if saved:

                st.success(
                    "✅ Attendance saved to GitHub."
                )

                st.write(
                    f"🕐 Attendance time: "
                    f"**{record['time']}**"
                )

        except Exception as e:

            st.error(
                "❌ Face recognition failed."
            )

            st.exception(e)


# =========================================================
# ATTENDANCE SHEET
# =========================================================

elif menu == "Attendance Sheet":

    st.title(
        "📊 Attendance Sheet"
    )

    try:

        attendance = load_attendance()

        if attendance.empty:

            st.info(
                "No attendance records."
            )

            st.stop()

        col1, col2 = st.columns(2)

        with col1:

            selected_date = (
                st.date_input(
                    "Select Date",
                    value=get_now().date()
                )
            )

        with col2:

            search = st.text_input(
                "Search Name / Number / Position"
            ).strip().lower()

        date_string = (
            selected_date.strftime(
                "%Y-%m-%d"
            )
        )

        filtered = attendance[
            attendance["date"]
            .astype(str)
            == date_string
        ].copy()

        if search:

            mask = (

                filtered["name"]
                .astype(str)
                .str.lower()
                .str.contains(
                    search,
                    na=False
                )

                |

                filtered["number"]
                .astype(str)
                .str.lower()
                .str.contains(
                    search,
                    na=False
                )

                |

                filtered["position"]
                .astype(str)
                .str.lower()
                .str.contains(
                    search,
                    na=False
                )
            )

            filtered = filtered[
                mask
            ]

        st.dataframe(

            filtered[
                [
                    "number",
                    "name",
                    "position",
                    "date",
                    "time",
                    "status"
                ]
            ],

            use_container_width=True,

            hide_index=True
        )

        csv_data = (
            filtered
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(

            "⬇️ Download CSV",

            data=csv_data,

            file_name=(
                f"attendance_"
                f"{date_string}.csv"
            ),

            mime="text/csv"
        )

    except Exception as e:

        st.error(
            str(e)
        )


# =========================================================
# PEOPLE DATABASE
# =========================================================

elif menu == "People Database":

    st.title(
        "👥 People Database"
    )

    try:

        people = load_people()

        if people.empty:

            st.info(
                "No registered people."
            )

            st.stop()

        st.dataframe(

            people[
                [
                    "number",
                    "name",
                    "position",
                    "created_at"
                ]
            ],

            use_container_width=True,

            hide_index=True
        )

    except Exception as e:

        st.error(
            str(e)
        )
```
