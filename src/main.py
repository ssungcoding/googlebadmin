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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 비디오의 총 프레임 수
        section_frames = total_frames // 4  # 각 섹션의 프레임 수

        hand_movement_count = [0, 0, 0, 0]  # 각 섹션에서 감지된 손동작 횟수

        with mp_pose.Pose(
            static_image_mode=False, min_detection_confidence=0.5
        ) as pose:
            section_index = 0  # 현재 섹션 인덱스
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logging.info("비디오 프레임을 더 이상 읽을 수 없습니다.")
                    break

                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # 현재 프레임에서 손동작이 감지되었는지 확인
                if results.pose_landmarks and is_badminton_motion(
                    results.pose_landmarks
                ):
                    hand_movement_count[section_index] += 1

                # 현재 섹션의 마지막 프레임에 도달했는지 확인
                if (
                    cap.get(cv2.CAP_PROP_POS_FRAMES)
                    >= (section_index + 1) * section_frames
                ):
                    section_index += 1  # 다음 섹션으로 이동
                    if section_index >= 4:  # 모든 섹션을 검사했으면 종료
                        break

        cap.release()
        logging.info(f"손동작 감지 횟수: {hand_movement_count}")

        # 각 섹션에서 손동작이 2번 이상 감지되었는지 확인
        section_results = [count >= 2 for count in hand_movement_count]
        logging.info(f"각 섹션의 손동작 감지 결과: {section_results}")

        # True가 3개 이상인 경우 최종 결과를 True로 설정
        final_result = section_results.count(True) >= 3
        logging.info(f"최종 결과: {final_result}")

        return {"pose_detected": final_result}
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
