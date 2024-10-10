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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 비디오의 총 프레임 수
    section_frames = total_frames // 4  # 각 섹션의 프레임 수

    hand_movement_count = [0, 0, 0, 0]  # 각 섹션에서 감지된 손동작 횟수
    section_index = 0  # 현재 섹션 인덱스

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 현재 프레임에서 손동작이 감지되었는지 확인
            if results.pose_landmarks and is_badminton_motion(results.pose_landmarks):
                hand_movement_count[section_index] += 1

            # 현재 섹션의 마지막 프레임에 도달했는지 확인
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= (section_index + 1) * section_frames:
                section_index += 1  # 다음 섹션으로 이동
                if section_index >= 4:  # 모든 섹션을 검사했으면 종료
                    break

    cap.release()

    # 각 섹션에서 손동작이 2번 이상 감지되었는지 확인
    section_results = [count >= 2 for count in hand_movement_count]

    # True가 3개 이상인 경우 최종 결과를 True로 설정
    final_result = section_results.count(True) >= 3

    return {"pose_detected": final_result}
