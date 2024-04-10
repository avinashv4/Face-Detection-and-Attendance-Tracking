import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import time
import dlib 

last_recognized = {}

def reset_all():
    confirmation = input("Are you sure you want to reset all data? (y/no): ").lower()

    if confirmation == 'y':
        # Reset 'database' folder
        database_folder = 'database'
        for filename in os.listdir(database_folder):
            file_path = os.path.join(database_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        # Reset attendance Excel sheet (assuming 'AttendanceDB.xlsx' is your file)
        excel_file_path = 'AttendanceDB.xlsx'
        wb = openpyxl.load_workbook(excel_file_path)
        sheet = wb.active

        # Remove all rows except the one containing 'userid'
        for row in range(sheet.max_row, 1, -1):
            if sheet.cell(row=row, column=1).value != 'userid':
                sheet.delete_rows(row)

        # Save the modified workbook
        wb.save(excel_file_path)

        print("Reset completed.")
    else:
        print("Reset canceled.")

def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.yml")


def get_next_user_id(data_sheet):
    max_id = data_sheet['UserID'].max()
    if pd.isnull(max_id):
        return 1
    else:
        return int(max_id) + 1
    
def generate_dataset(img, id, img_id):
    if img is not None:
        img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        target_width = 100
        target_height = 100
        img_gray_resized = cv2.resize(img_gray, (target_width, target_height))
        img_gray_blurred = cv2.GaussianBlur(img_gray_resized, (5, 5), 0)
        img_gray_equalized = cv2.equalizeHist(img_gray_blurred)
        file_path="database/user." + str(id) + "." + str (img_id) + ".jpg"
        cv2.imwrite(file_path, img_gray_equalized)

def create_new_student(data_sheet, img_id):
    username = input("Enter the new student's name: ")
    user_id = get_next_user_id(data_sheet)

    new_data = pd.DataFrame({'UserID': [user_id], 'UserName': [username]})
    data_sheet = pd.concat([data_sheet, new_data], ignore_index=True)
    data_sheet.to_excel("AttendanceDB.xlsx", index=False)

    start_time = time.time()
    frames_captured=0
    while frames_captured<num_frames_to_capture:
        _, img = video_capture.read()
        img,roi_img = detect(img, faceCascade, img_id, user_id,username)
        cv2.imshow("Face Detection", img)
        cv2.waitKey(1)  
        
        generate_dataset(roi_img, user_id, img_id)
        img_id += 1\
        
        frames_captured+=1

    video_capture.release()
    cv2.destroyAllWindows()

    train_classifier("database")
    print("New Student has been added")

def detect (img, faceCascade, img_id, user_id,username):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords, roi_img = draw_boundary_for_detect(img, faceCascade, 1.1, 10, color['blue'],"Face",clf, data_sheet,username) 

    return img,roi_img

def draw_boundary_for_detect(img, classifier, scaleFactor, minNeighbours, color, text, clf, user_id,username):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords = []
    roi_img = None

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        roi_gray = gray_img[y:y + h, x:x + w]

        eye_cascade = cv2.CascadeClassifier('hcc_eye.xml')
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes)>=2:
            cv2.putText(img, username, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            coords = [x, y, w, h]
            roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]

    return coords, roi_img

def draw_boundary_for_recognize(img, classifier, scaleFactor, minNeighbours, color, text, clf, data_sheet):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    
    recognized_user_ids = []
    coords=[]
    confidence_threshold = 65
    min_detections = 50  # Minimum number of detections required
    
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        roi_gray = gray_img[y:y + h, x:x + w]
        id, confidence = clf.predict(roi_gray)

        if confidence > confidence_threshold:
            user_id = int(id)

            if not data_sheet.empty and any(data_sheet['UserID'] == user_id):
                username = data_sheet.loc[data_sheet['UserID'] == user_id, 'UserName'].values[0]
            else:
                username = "Unknown User"

            if user_id in last_recognized:
                last_recognized[user_id] += 1
                if last_recognized[user_id] >= min_detections:
                    recognized_user_ids.append(user_id)
                    
                    text_to_display = f"{username}(marked present)"
                    cv2.putText(img, text_to_display, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                last_recognized[user_id] = 1

            if user_id not in recognized_user_ids:
                cv2.putText(img, f"{username} ({confidence:.2f})", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            coords=[x,y,w,h]

    return recognized_user_ids,coords

def recognize(img, clf, faceCascade, data_sheet,date):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    
    recognized_user_ids,coords = draw_boundary_for_recognize(img, faceCascade, 1.1, 10, color["white"], "Face", clf, data_sheet)

    for user_id in recognized_user_ids:
        
            update_attendance(data_sheet, user_id, date, 'Present')


    return img

def update_attendance(data_sheet, user_id, date, status):
    if data_sheet.loc[data_sheet['UserID'] == user_id, date].values[0] != 'Present':
        data_sheet.loc[data_sheet['UserID'] == user_id, date] = status


eye_cascade = cv2.CascadeClassifier('hcc_eye.xml')
faceCascade = cv2.CascadeClassifier("hcc_frontal.xml")

clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

video_capture = cv2.VideoCapture(0)
data_sheet = pd.read_excel("AttendanceDB.xlsx")

img_id = 0

while True:
    print("Choose operation:")
    print("1. Create new student and generate dataset")
    print("2. Take attendance")
    print("3. Open attendance sheet")
    print("4. Quit")

    choice = input("Enter your choice (1-4): ")

    if choice == '1':
        _, img = video_capture.read()
        num_frames_to_capture=250
        create_new_student(data_sheet, img_id)

    elif choice == '2':
        date = input("Enter the date (YYYY-MM-DD): ")
        if date not in data_sheet.columns:
            data_sheet[date] = ''
        while True:
            _, img = video_capture.read()
            ret, img = video_capture.read()

            if not ret:
                print("Error: Failed to capture a frame from the video stream.")
                break
            img= recognize(img, clf, faceCascade, data_sheet, date)
            cv2.imshow("Face Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                print("Attendance Closed!")
                break
        remaining_ids = data_sheet.loc[(data_sheet[date] != 'Present'), 'UserID']
        for user_id in remaining_ids:
            update_attendance(data_sheet, user_id, date, 'Absent')
        data_sheet.to_excel("AttendanceDB.xlsx", index=False)

    elif choice == '3':
        os.system("start excel.exe AttendanceDB.xlsx")
        
    elif choice == '4':
        break

    elif choice == 'reset':
        reset_all();
    
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")

video_capture.release()
cv2.destroyAllWindows()



