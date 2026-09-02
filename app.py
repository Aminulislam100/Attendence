import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os
from github import Github

# ==========================================
# 1. Page Config & File Constants
# ==========================================
st.set_page_config(page_title="Smart Attendance System", page_icon="🧑‍💻", layout="wide")

DB_DIR = "database_faces"
ATTENDANCE_FILE = "attendance_sheet.csv"

os.makedirs(DB_DIR, exist_ok=True)

if not os.path.exists(ATTENDANCE_FILE):
    df_init = pd.DataFrame(columns=["Name", "Number", "Position", "Date", "Time"])
    df_init.to_csv(ATTENDANCE_FILE, index=False)

# ==========================================
# 2. GitHub Persistent Auto-Sync Function
# ==========================================
def sync_to_github(file_path, commit_message):
    """GitHub Repositories-e file auto push/update kore"""
    if "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets:
        try:
            g = Github(st.secrets["GITHUB_TOKEN"])
            repo = g.get_repo(st.secrets["GITHUB_REPO"])
            
            with open(file_path, "rb") as f:
                content = f.read()
                
            try:
                # Update existing file
                contents = repo.get_contents(file_path)
                repo.update_file(contents.path, commit_message, content, contents.sha)
            except Exception:
                # Create new file if doesn't exist
                repo.create_file(file_path, commit_message, content)
            return True
        except Exception as e:
            st.warning(f"⚠️ GitHub sync error: {e}")
            return False
    return False

# ==========================================
# 3. Helper Functions
# ==========================================
def load_known_faces():
    known_encodings = []
    known_details = []
    
    if os.path.exists(DB_DIR):
        for file in os.listdir(DB_DIR):
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                parts = os.path.splitext(file)[0].split('_')
                if len(parts) >= 3:
                    person_name = parts[0]
                    person_number = parts[1]
                    person_position = "_".join(parts[2:])
                    
                    img_path = os.path.join(DB_DIR, file)
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_details.append({
                            "Name": person_name,
                            "Number": person_number,
                            "Position": person_position
                        })
    return known_encodings, known_details

def save_attendance(person_info):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%I:%M:%S %p")
    
    df = pd.read_csv(ATTENDANCE_FILE)
    already_marked = df[(df['Name'] == person_info['Name']) & (df['Date'] == current_date)]
    
    if not already_marked.empty:
        return False, f"⚠️ {person_info['Name']}, apnar ajker hajira agei newa hoyeche!"
    
    new_entry = pd.DataFrame([{
        "Name": person_info["Name"],
        "Number": person_info["Number"],
        "Position": person_info["Position"],
        "Date": current_date,
        "Time": current_time
    }])
    
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
    
    # Push updated sheet to GitHub
    sync_to_github(ATTENDANCE_FILE, f"Update attendance: {person_info['Name']}")
    
    return True, f"✅ Hajira Shofol! Name: {person_info['Name']} | Time: {current_time}"

# ==========================================
# 4. User Interface
# ==========================================
st.title("🧑‍💻 Smart Auto Attendance System")

tab1, tab2, tab3 = st.tabs(["👤 Nuton Entry", "📸 Hajira Din", "📊 Attendance Sheet"])

with tab1:
    st.header("Nuton Person Profile Registration")
    c1, c2 = st.columns([1, 1])
    with c1:
        name = st.text_input("Name")
        number = st.text_input("Mobile Number")
        position = st.text_input("Position")
    with c2:
        reg_cam = st.camera_input("Face Scan Korun", key="reg_cam")
        
    if st.button("💾 Save to Database & GitHub", type="primary"):
        if name and number and position and reg_cam:
            c_name = name.strip().replace(" ", "-")
            c_num = number.strip().replace(" ", "")
            c_pos = position.strip().replace(" ", "-")
            
            filename = f"{c_name}_{c_num}_{c_pos}.jpg"
            filepath = os.path.join(DB_DIR, filename)
            
            with open(filepath, "wb") as f:
                f.write(reg_cam.getbuffer())
                
            # GitHub repo-te photo push
            sync_to_github(filepath, f"Add new profile photo: {c_name}")
            st.success("🎉 Profile database and GitHub repo-te save hoye geche!")
            st.rerun()
        else:
            st.error("⚠️ Shobgula input ghor puron korun.")

with tab2:
    st.header("Live Attendance Scan")
    known_encodings, known_details = load_known_faces()
    
    if not known_encodings:
        st.warning("⚠️ Database-e kono profile nei.")
    else:
        att_cam = st.camera_input("Camera-y takan", key="att_cam")
        if att_cam:
            bytes_data = att_cam.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_img)
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            
            if not face_encodings:
                st.error("❌ Mukh shonakhto kora jayni!")
            else:
                matched = False
                for encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.48)
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    
                    if True in matches:
                        best_idx = np.argmin(distances)
                        person = known_details[best_idx]
                        ok, msg = save_attendance(person)
                        if ok:
                            st.success(msg)
                        else:
                            st.warning(msg)
                        st.json({
                            "Name": person["Name"].replace("-", " "),
                            "Number": person["Number"],
                            "Position": person["Position"].replace("-", " ")
                        })
                        matched = True
                        break
                if not matched:
                    st.error("❌ Face match korche na!")

with tab3:
    st.header("📊 Live Attendance Sheet")
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "attendance.csv", "text/csv")
