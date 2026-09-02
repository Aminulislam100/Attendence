import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ==========================================
# 1. Folder and File Setup
# ==========================================
DB_DIR = "database_faces"
ATTENDANCE_FILE = "attendance_sheet.csv"

os.makedirs(DB_DIR, exist_ok=True)

if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Number", "Position", "Date", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

st.set_page_config(page_title="Smart Attendance System", layout="wide", page_icon="🧑‍💻")
st.title("🧑‍💻 Automated Face Attendance System")

tab1, tab2 = st.tabs(["📝 Nuton Manush Entry (Registration)", "📸 Live Hajira (Attendance)"])

# ==========================================
# 2. Tab 1: Nuton Manush Entry
# ==========================================
with tab1:
    st.header("Nuton Manusher Database Toiri Korun")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Byaktir Tathya Din:**")
        name = st.text_input("Name")
        number = st.text_input("Mobile Number")
        position = st.text_input("Position")
    
    with col2:
        st.write("**Chobi Tulun:**")
        img_file = st.camera_input("Database er jonno chobi din", key="register_cam")
        
    if st.button("💾 Database e Save Korun", type="primary", use_container_width=True):
        if name and number and position and img_file:
            file_name = f"{name}_{number}_{position}.jpg"
            file_path = os.path.join(DB_DIR, file_name)
            
            with open(file_path, "wb") as f:
                f.write(img_file.getbuffer())
            st.success(f"✅ {name}-er profile shofolbhabe database e save hoyeche!")
            st.rerun()
        else:
            st.error("⚠️ Name, Number, Position ebong Chobi—shobgula input dewa baddhotamulok!")

# ==========================================
# Function: Load Known Faces
# ==========================================
def load_known_faces():
    known_encodings = []
    known_details = []
    
    if os.path.exists(DB_DIR):
        for file in os.listdir(DB_DIR):
            if file.endswith(('jpg', 'jpeg', 'png')):
                details = file.split('.')[0].split('_')
                if len(details) == 3:
                    img_path = os.path.join(DB_DIR, file)
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_details.append({
                            "Name": details[0],
                            "Number": details[1],
                            "Position": details[2]
                        })
    return known_encodings, known_details

# ==========================================
# 3. Tab 2: Live Hajira and Sheet Update
# ==========================================
with tab2:
    st.header("Hajira Din (Face Recognition)")
    
    known_encodings, known_details = load_known_faces()
    
    if len(known_encodings) == 0:
        st.warning("⚠️ Database e kono manusher chobi nei. Age 'Nuton Manush Entry' theke profile toiri korun.")
    else:
        attend_cam = st.camera_input("Hajira dite camera-y takan", key="attendance_cam")
        
        if attend_cam:
            with st.spinner("Face melano hocche..."):
                bytes_data = attend_cam.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_img)
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                
                if len(face_encodings) == 0:
                    st.error("⚠️ Chobite kono mukh shonakhto kora jayni! Arokbar bhalobhabe chobi tulun.")
                else:
                    for encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
                        face_distances = face_recognition.face_distance(known_encodings, encoding)
                        
                        if True in matches:
                            best_match_index = np.argmin(face_distances)
                            matched_person = known_details[best_match_index]
                            
                            now = datetime.now()
                            dt_string = now.strftime("%Y-%m-%d")
                            tm_string = now.strftime("%I:%M:%S %p")
                            
                            df = pd.read_csv(ATTENDANCE_FILE)
                            already_present = df[(df['Name'] == matched_person["Name"]) & (df['Date'] == dt_string)]
                            
                            if not already_present.empty:
                                st.info(f"👍 {matched_person['Name']}, apnar ajker hajira agei newa hoyeche!")
                            else:
                                new_record = pd.DataFrame([{
                                    "Name": matched_person["Name"],
                                    "Number": matched_person["Number"],
                                    "Position": matched_person["Position"],
                                    "Date": dt_string,
                                    "Time": tm_string
                                }])
                                df = pd.concat([df, new_record], ignore_index=True)
                                df.to_csv(ATTENDANCE_FILE, index=False)
                                
                                st.success(f"✅ Hajira shofol hoyeche! Swagotom {matched_person['Name']} ({matched_person['Position']})")
                        else:
                            st.error("❌ Mukh database er karor shathe milche na! Unknown byakti.")

    # ==========================================
    # 4. Live Sheet Display
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Protidiner Hajirar Sheet (Attendance Sheet)")
    
    try:
        df_display = pd.read_csv(ATTENDANCE_FILE)
        if not df_display.empty:
            st.dataframe(df_display, width=1200, height=400)
            
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Sheet Download Korun (CSV)",
                data=csv,
                file_name="attendance_records.csv",
                mime="text/csv",
            )
        else:
            st.write("Sheet e ekhono kono data nei.")
    except Exception as e:
        st.write("Sheet load kora jacche na.")
