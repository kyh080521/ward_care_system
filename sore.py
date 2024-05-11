import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)
cv2.namedWindow("Eye Tracking")

# 화면 크기
screen_width, screen_height = 640, 480

# 점 초기 위치 (화면 중심)
point_x, point_y = screen_width // 2, screen_height // 2

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_landmarks = [face_landmarks.landmark[i] for i in range(159, 145, -1)]  # 눈 주변 랜드마크
            right_eye_landmarks = [face_landmarks.landmark[i] for i in range(386, 380, -1)]  # 눈 주변 랜드마크
            left_eye_center = (left_eye_landmarks[0].x, left_eye_landmarks[0].y)  # 왼쪽 눈동자 랜드마크
            right_eye_center = (right_eye_landmarks[0].x, right_eye_landmarks[0].y)  # 오른쪽 눈동자 랜드마크

            # 눈동자 위치로부터 눈 주변 랜드마크까지의 상대적인 거리 계산
            left_eye_distance_x = left_eye_landmarks[4].x - left_eye_center[0]
            left_eye_distance_y = left_eye_landmarks[4].y - left_eye_center[1]
            right_eye_distance_x = right_eye_center[0] - right_eye_landmarks[4].x
            right_eye_distance_y = right_eye_center[1] - right_eye_landmarks[4].y

            # 눈동자 이동량에 따라 방향 결정
            if abs(left_eye_distance_x) > abs(right_eye_distance_x):
                horizontal_direction = "Look Left" if left_eye_distance_x > 0 else "Look Right"
            else:
                horizontal_direction = "Look Left" if right_eye_distance_x < 0 else "Look Right"

            if abs(left_eye_distance_y) > abs(right_eye_distance_y):
                vertical_direction = "Look Up" if left_eye_distance_y > 0 else "Look Down"
            else:
                vertical_direction = "Look Up" if right_eye_distance_y > 0 else "Look Down"

            cv2.putText(image, horizontal_direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, vertical_direction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 화면 중심을 기준으로 점 이동
            if horizontal_direction == "Look Left":
                point_x -= 10
            elif horizontal_direction == "Look Right":
                point_x += 10

            if vertical_direction == "Look Up":
                point_y -= 10
            elif vertical_direction == "Look Down":
                point_y += 10

            # 화면 중심을 기준으로 십자 모양 이동 제한
            if abs(point_x - screen_width // 2) > abs(point_y - screen_height // 2):
                point_y = screen_height // 2
            else:
                point_x = screen_width // 2

            # 화면 끝까지만 이동 가능하도록 제한
            point_x = max(0, min(point_x, screen_width))
            point_y = max(0, min(point_y, screen_height))

            cv2.circle(image, (point_x, point_y), 5, (0, 255, 0), -1)

    cv2.imshow("Eye Tracking", image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
