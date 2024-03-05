import cv2
import face_recognition
import numpy as np

# Initializes the camera and runs the face recognition process
def run_face_recognition():
    camera = cv2.VideoCapture(0)
    reference_face_encodings, reference_face_names = load_reference_face()

    analyze_frame_flag = True

    while cv2.waitKey(1) & 0xFF != ord("q"):
        _, current_frame = camera.read()

        if analyze_frame_flag:
            face_locations, face_names, face_distances = analyze_frame(current_frame, reference_face_encodings, reference_face_names)
            display_result(current_frame, face_locations, face_names, face_distances)
            cv2.imshow("FaceLink", current_frame)

        analyze_frame_flag = not analyze_frame_flag

    camera.release()
    cv2.destroyAllWindows()

# Loads the reference face image and its associated person's name
def load_reference_face():
    image_path = "areeb.jpg"
    person_name = "Areeb"

    reference_image = face_recognition.load_image_file(image_path)
    reference_face_encoding = face_recognition.face_encodings(reference_image)[0]

    return [reference_face_encoding], [person_name]

# Analyzes the current frame for faces and compares with the reference face
def analyze_frame(current_frame, reference_face_encodings, reference_face_names):
    resized_frame = cv2.resize(current_frame, (0, 0), fx = 0.25, fy = 0.25)
    resized_rgb_frame = resized_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(resized_rgb_frame)
    face_encodings = face_recognition.face_encodings(resized_rgb_frame, face_locations)

    face_names = ["Unknown"] * len(face_locations)
    face_distances = np.zeros(len(face_locations))

    for i, (face_encoding, location) in enumerate(zip(face_encodings, face_locations)):
        face_matched = face_recognition.compare_faces(reference_face_encodings, face_encoding)
        best_match_index = np.argmin(face_recognition.face_distance(reference_face_encodings, face_encoding))

        if face_matched[best_match_index]:
            face_names[i] = reference_face_names[best_match_index]
            face_distances[i] = face_recognition.face_distance([reference_face_encodings[best_match_index]], face_encoding)[0]

    return face_locations, face_names, face_distances

# Displays rectangles around recognized faces with names and match accuracies
def display_result(current_frame, face_locations, face_names, face_distances):
    for (top, right, bottom, left), recognized_name, match_distance in zip(face_locations, face_names, face_distances):
        top, right, bottom, left = [4 * resize_factor for resize_factor in (top, right, bottom, left)]

        if recognized_name == "Unknown":
            box_color = (0, 0, 255)
            text_color = (0, 0, 0)
        else:
            box_color = (0, 255, 0)
            text_color = (0, 0, 0)

        cv2.rectangle(current_frame, (left, top), (right, bottom), box_color, 2)

        if recognized_name == "Unknown":
            match_accuracy = 0
        else:
            match_accuracy = 100 - (match_distance * 100)

        display_text = f'{recognized_name} ({match_accuracy:.2f})%'

        cv2.rectangle(current_frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        cv2.putText(current_frame, display_text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

run_face_recognition()