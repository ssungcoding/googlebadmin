# Python 3.10을 사용하여 기본 이미지를 설정합니다.
FROM python:3.10-slim

# 필요한 시스템 종속성 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgtk2.0-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리를 설정합니다.
WORKDIR /app

# 애플리케이션의 종속성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# FastAPI 앱 실행
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
