from fastapi import FastAPI, BackgroundTasks, File, UploadFile
import os
import mediapipe as mp
import cv2
import logging

app = FastAPI()

# 로그 설정
logging.basicConfig(level=logging.INFO)


def is_badminton_motion(pose_landmarks):
    left_hand = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    right_hand = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

    if left_hand and right_hand:
        return True
    return False


def process_video(file_location: str):
    try:
        cap = cv2.VideoCapture(file_location)
        if not cap.isOpened():
            logging.error(f"비디오 파일 열기 실패: {file_location}")
            return {"error": "비디오 파일을 열 수 없습니다."}

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=False, min_detection_confidence=0.5
        ) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logging.info("비디오 프레임을 더 이상 읽을 수 없습니다.")
                    break
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks and is_badminton_motion(
                    results.pose_landmarks
                ):
                    logging.info("배드민턴 동작 감지됨")
                    return {"pose_detected": True}  # 배드민턴 동작 감지됨

        cap.release()
        logging.info("배드민턴 동작 감지되지 않음")
        return {"pose_detected": False}  # 배드민턴 동작 감지되지 않음
    finally:
        # 비디오 파일 삭제
        if os.path.exists(file_location):
            os.remove(file_location)
            logging.info(f"임시 파일 삭제 완료: {file_location}")


@app.post("/analyze-video/")
async def analyze_video(
    background_tasks: BackgroundTasks, video: UploadFile = File(...)
):
    file_location = f"/tmp/{video.filename}"
    with open(file_location, "wb") as file_object:
        file_object.write(video.file.read())

    logging.info(f"비디오 파일 수신 및 저장 완료: {file_location}")

    # 비디오 분석 작업을 백그라운드에서 실행
    background_tasks.add_task(process_video, file_location)

    return {"status": "비디오 분석 작업이 시작되었습니다. 결과를 기다려 주세요."}
