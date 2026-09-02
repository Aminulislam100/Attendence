import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ==========================================
# 1. Page Config & Directory Setup
# ==========================================
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🧑‍💻",
    layout="wide"
)

DB_DIR = "database_faces"
ATTENDANCE_FILE = "attendance_sheet.csv"

# Directory & CSV create if not exists
os.makedirs(DB_DIR, exist_ok=True)

if not os.path.exists(ATTENDANCE_FILE):
    df_init = pd.DataFrame(columns=["Name", "Number", "Position", "Date", "Time"])
    df_init.to_csv(ATTENDANCE_FILE, index=False)

# ==========================================
# 2. Helper Functions
# ==========================================
def load_known_faces():
    """Database folder theke shob image and metadata load kore"""
    known_encodings = []
    known_details = []
    
    if os.path.exists(DB_DIR):
        for file in os.listdir(DB_DIR):
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                # Filename structure: Name_Number_Position.jpg
                filename_without_ext = os.path.splitext(file)[0]
                parts = filename_without_ext.split('_')
                
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
    """Attendance CSV file-e record save kore"""
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%I:%M:%S %p")
    
    df = pd.read_csv(ATTENDANCE_FILE)
    
    # Check if already present today
    already_marked = df[(df['Name'] == person_info['Name']) & (df['Date'] == current_date)]
    
    if not already_marked.empty:
        return False, f"⚠️ {person_info['Name']}, apnar ajker ({current_date}) hajira agei newa hoyeche!"
    
    # Append new record
    new_entry = pd.DataFrame([{
        "Name": person_info["Name"],
        "Number": person_info["Number"],
        "Position": person_info["Position"],
        "Date": current_date,
        "Time": current_time
    }])
    
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
    
    return True, f"✅ Hajira Shofol! Name: {person_info['Name']} | Position: {person_info['Position']} | Time: {current_time}"

# ==========================================
# 3. Main UI Header
# ==========================================
st.title("🧑‍💻 Automated Face Attendance System")
st.markdown("Face recognition-er maddhome automatic attendance ebong database management.")

tab1, tab2, tab3 = st.tabs([
    "👤 Nuton Manush Entry (Registration)", 
    "📸 Hajira Din (Live Attendance)", 
    "📊 Attendance Sheet"
])

# ==========================================
# TAB 1: Nuton Person Registration
# ==========================================
with tab1:
    st.header("Nuton Manusher Database Entry")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Tathya Din")
        input_name = st.text_input("Full Name", placeholder="e.g. Aminul Islam")
        input_number = st.text_input("Mobile Number", placeholder="e.g. 01700000000")
        input_position = st.text_input("Position / Role", placeholder="e.g. Software Engineer")
    
    with col2:
        st.subheader("2. Face Image Capture Korun")
        reg_cam = st.camera_input("Database er jonno chobi tulun", key="reg_camera")
    
    if st.button("💾 Database-e Save Korun", type="primary", use_container_width=True):
        if not (input_name and input_number and input_position and reg_cam):
            st.error("⚠️ Shobgula input (Name, Number, Position ebong Chobi) dewa baddhotamulok!")
        else:
            # Clean spaces for filename
            clean_name = input_name.strip().replace(" ", "-")
            clean_number = input_number.strip().replace(" ", "")
            clean_position = input_position.strip().replace(" ", "-")
            
            filename = f"{clean_name}_{clean_number}_{clean_position}.jpg"
            filepath = os.path.join(DB_DIR, filename)
            
            with open(filepath, "wb") as f:
                f.write(reg_cam.getbuffer())
                
            st.success(f"🎉 {input_name}-er profile database folder-e save hoye geche!")
            st.rerun()

# ==========================================
# TAB 2: Live Attendance
# ==========================================
with tab2:
    st.header("Camera-y Takiyye Hajira Din")
    
    known_encodings, known_details = load_known_faces()
    
    if len(known_encodings) == 0:
        st.warning("⚠️ Database-e kono profile nei! Age 'Nuton Manush Entry' tab theke profile toiri korun.")
    else:
        st.info(f"📁 Database-e mot {len(known_encodings)} joner profile ache.")
        att_cam = st.camera_input("Hajira dite camera-y takan", key="att_camera")
        
        if att_cam:
            with st.spinner("Face scan kora hocche..."):
                bytes_data = att_cam.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_img)
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                
                if len(face_encodings) == 0:
                    st.error("❌ Mukh shonakhto kora jayni! Bhalobhabe camera-r samne ashun.")
                else:
                    matched = False
                    for encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.48)
                        face_distances = face_recognition.face_distance(known_encodings, encoding)
                        
                        if True in matches:
                            best_match_idx = np.argmin(face_distances)
                            person = known_details[best_match_idx]
                            
                            success, msg = save_attendance(person)
                            if success:
                                st.success(msg)
                                st.balloons()
                            else:
                                st.warning(msg)
                            
                            # Show matched info box
                            st.json({
                                "Name": person["Name"].replace("-", " "),
                                "Mobile Number": person["Number"],
                                "Position": person["Position"].replace("-", " ")
                            })
                            matched = True
                            break
                    
                    if not matched:
                        st.error("❌ Face match korche na! Apni database-e registered nen.")

# ==========================================
# TAB 3: Attendance Sheet
# ==========================================
with tab3:
    st.header("📊 Attendance Sheet (Live Record)")
    
    try:
        df_sheet = pd.read_csv(ATTENDANCE_FILE)
        
        if not df_sheet.empty:
            # Re-format display names
            df_display = df_sheet.copy()
            df_display['Name'] = df_display['Name'].str.replace('-', ' ')
            df_display['Position'] = df_display['Position'].str.replace('-', ' ')
            
            st.dataframe(df_display, use_container_width=True, height=400)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1:
                csv_data = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Sheet (CSV)",
                    data=csv_data,
                    file_name=f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv",
                    type="primary"
                )
        else:
            st.info("Ekhono kono hajira record hoyni.")
    except Exception as e:
        st.error(f"Sheet load korte shomoshya hoyeche: {e}")
