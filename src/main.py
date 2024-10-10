from fastapi import FastAPI, File, UploadFile
import os
import mediapipe as mp
import cv2
import asyncio

app = FastAPI()


def is_badminton_motion(pose_landmarks):
    # 배드민턴 동작을 인식하기 위한 조건 로직을 추가
    left_hand = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    right_hand = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

    if left_hand and right_hand:  # 예시: 손 위치가 특정 범위 내에 있는 경우
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
    motion_count = 0  # 모션 카운트를 위한 변수 초기화
    frame_skip = 5  # 프레임을 5개씩 건너뜁니다 (예시값, 필요에 따라 조정 가능)

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as pose:
        frame_index = 0  # 현재 프레임 인덱스를 추적
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip == 0:  # 스킵된 프레임만 처리
                frame = cv2.resize(
                    frame, (640, 480)
                )  # 프레임 크기를 줄여 처리 속도 향상
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks and is_badminton_motion(
                    results.pose_landmarks
                ):
                    motion_count += 1
            frame_index += 1

    cap.release()
    return {
        "pose_detected": motion_count >= 200
    }  # 최종적으로 모션이 200회 이상 감지되었는지 반환
