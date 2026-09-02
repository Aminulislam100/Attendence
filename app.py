import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ==========================================
# ১. ফোল্ডার ও ফাইল সেটআপ
# ==========================================
DB_DIR = "database_faces"
ATTENDANCE_FILE = "attendance_sheet.csv"

# ডেটাবেস ফোল্ডার না থাকলে তৈরি করবে
os.makedirs(DB_DIR, exist_ok=True)

# হাজিরা শিট (CSV) না থাকলে তৈরি করবে
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Number", "Position", "Date", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

st.set_page_config(page_title="Smart Attendance System", layout="wide", page_icon="🧑‍💻")
st.title("🧑‍💻 Automated Face Attendance System")

# ট্যাবের মাধ্যমে দুই দিকের কাজ ভাগ করা
tab1, tab2 = st.tabs(["📝 নতুন মানুষ এন্ট্রি (Registration)", "📸 লাইভ হাজিরা (Attendance)"])

# ==========================================
# ২. Tab 1: নতুন মানুষ এন্ট্রি (Registration)
# ==========================================
with tab1:
    st.header("নতুন মানুষের ডেটাবেস তৈরি করুন")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**ব্যক্তির তথ্য দিন:**")
        name = st.text_input("নাম (Name)")
        number = st.text_input("মোবাইল নম্বর (Number)")
        position = st.text_input("পদবি (Position)")
    
    with col2:
        st.write("**ছবি তুলুন:**")
        img_file = st.camera_input("ডেটাবেসের জন্য ছবি দিন", key="register_cam")
        
    if st.button("💾 ডেটাবেসে সেভ করুন", type="primary", use_container_width=True):
        if name and number and position and img_file:
            # ফাইলের নাম হিসেবে ব্যক্তির তথ্য সেভ করা হচ্ছে
            file_name = f"{name}_{number}_{position}.jpg"
            file_path = os.path.join(DB_DIR, file_name)
            
            with open(file_path, "wb") as f:
                f.write(img_file.getbuffer())
            st.success(f"✅ {name}-এর প্রোফাইল সফলভাবে ডেটাবেসে সেভ হয়েছে!")
        else:
            st.error("⚠️ নাম, নাম্বার, পদবি এবং ছবি—সবগুলো ইনপুট দেওয়া বাধ্যতামূলক!")

# ==========================================
# ফাংশন: ফোল্ডার থেকে ফেস ডেটা লোড করা
# ==========================================
@st.cache_data(ttl=60) # প্রতি ১ মিনিটে রিফ্রেশ হবে যাতে নতুন ডেটা পায়
def load_known_faces():
    known_encodings = []
    known_details = []
    
    for file in os.listdir(DB_DIR):
        if file.endswith(('jpg', 'jpeg', 'png')):
            # ফাইলের নাম (Name_Number_Position.jpg) থেকে ডিটেইলস আলাদা করা
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
# ৩. Tab 2: লাইভ হাজিরা ও শিট আপডেট
# ==========================================
with tab2:
    st.header("হাজিরা দিন (Face Recognition)")
    
    known_encodings, known_details = load_known_faces()
    
    if len(known_encodings) == 0:
        st.warning("⚠️ ডেটাবেসে কোনো মানুষের ছবি নেই। আগে 'নতুন মানুষ এন্ট্রি' থেকে প্রোফাইল তৈরি করুন।")
    else:
        attend_cam = st.camera_input("হাজিরা দিতে ক্যামেরায় তাকান", key="attendance_cam")
        
        if attend_cam:
            with st.spinner("ফেস মেলানো হচ্ছে..."):
                # ছবি প্রসেসিং
                bytes_data = attend_cam.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                
                # ফেস এনকোডিং বের করা
                face_locations = face_recognition.face_locations(rgb_img)
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                
                if len(face_encodings) == 0:
                    st.error("⚠️ ছবিতে কোনো মুখ শনাক্ত করা যায়নি! আরেকবার ভালোভাবে ছবি তুলুন।")
                else:
                    for encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
                        face_distances = face_recognition.face_distance(known_encodings, encoding)
                        
                        if True in matches:
                            best_match_index = np.argmin(face_distances)
                            matched_person = known_details[best_match_index]
                            
                            # বর্তমান সময় এবং তারিখ
                            now = datetime.now()
                            dt_string = now.strftime("%Y-%m-%d")
                            tm_string = now.strftime("%I:%M:%S %p")
                            
                            # CSV ফাইলে আপডেট
                            df = pd.read_csv(ATTENDANCE_FILE)
                            
                            # আজকে আগে হাজিরা দিয়েছে কি না চেক করা
                            already_present = df[(df['Name'] == matched_person["Name"]) & (df['Date'] == dt_string)]
                            
                            if not already_present.empty:
                                st.info(f"👍 {matched_person['Name']}, আপনার আজকের হাজিরা আগেই নেওয়া হয়েছে!")
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
                                
                                st.success(f"✅ হাজিরা সফল হয়েছে! স্বাগতম {matched_person['Name']} ({matched_person['Position']})")
                        else:
                            st.error("❌ মুখ ডেটাবেসের কারও সাথে মিলছে না! আননোন ব্যক্তি।")

    # ==========================================
    # ৪. লাইভ শিট ডিসপ্লে
    # ==========================================
    st.markdown("---")
    st.subheader("📊 প্রতিদিনের হাজিরার শিট (Attendance Sheet)")
    
    try:
        df_display = pd.read_csv(ATTENDANCE_FILE)
        # টেবিল রিফ্রেশ করার জন্য ছোট্ট একটি লজিক
        if not df_display.empty:
            st.dataframe(df_display, width=1200, height=400)
            
            # ডাউনলোড বাটন
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 শিট ডাউনলোড করুন (CSV)",
                data=csv,
                file_name="attendance_records.csv",
                mime="text/csv",
            )
        else:
            st.write("শিটে এখনও কোনো ডেটা নেই।")
    except Exception as e:
        st.write("শিট লোড করা যাচ্ছে না।")
