from fastapi import FastAPI, File, UploadFile
import os
import mediapipe as mp
import cv2

app = FastAPI()


def is_badminton_motion(pose_landmarks):
    # 배드민턴 동작을 인식하기 위한 조건 로직을 추가
    left_hand = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    right_hand = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

    if left_hand and right_hand:  # 조건 예시: 손의 위치가 특정 범위 내에 있을 때
        return True
    return False


@app.post("/analyze-video/")
async def analyze_video(video: UploadFile = File(...)):
    file_location = f"/tmp/{video.filename}"
    with open(file_location, "wb") as file_object:
        file_object.write(video.file.read())

    # 비디오 파일을 Mediapipe로 분석
    cap = cv2.VideoCapture(file_location)
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks and is_badminton_motion(results.pose_landmarks):
                return {"pose_detected": True}  # 배드민턴 동작 감지됨
    cap.release()
    return {"pose_detected": False}  # 배드민턴 동작 감지되지 않음
