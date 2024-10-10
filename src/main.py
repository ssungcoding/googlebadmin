from fastapi import FastAPI, File, UploadFile
import os
import subprocess
from src.mediapipe_JSON import generate_MP_JSON

app = FastAPI()


@app.post("/analyze-video/")
async def analyze_video(video: UploadFile = File(...)):
    file_location = f"/tmp/{video.filename}"
    with open(file_location, "wb") as file_object:
        file_object.write(video.file.read())

    # Call the Mediapipe function to process the video
    try:
        generate_MP_JSON({"files": {"test_img_path": file_location}})
        return {"message": "Video analyzed successfully."}
    except Exception as e:
        return {"error": str(e)}


@app.post("/plot-json/")
async def plot_json(json_file: UploadFile = File(...)):
    json_location = f"/tmp/{json_file.filename}"
    output_image = json_location.replace(".json", "_output.png")
    with open(json_location, "wb") as file_object:
        file_object.write(json_file.file.read())

    # Use subprocess to call plot_json.py
    try:
        subprocess.run(
            ["python3", "src/plot_json.py", json_location, output_image, "1000", "1000"]
        )
        return {"message": "Plot created successfully", "image_path": output_image}
    except Exception as e:
        return {"error": str(e)}
