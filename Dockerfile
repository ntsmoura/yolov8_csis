FROM ultralytics/ultralytics:latest-cpu

WORKDIR /yolov8_csis

RUN python3 -m pip install poetry==1.5.1

ENV PIP_NO_CACHE_DIR=false

RUN poetry config virtualenvs.create false

COPY pyproject.toml /yolov8_csis/pyproject.toml

RUN poetry install --without dev

COPY . /yolov8_csis

ENV PYTHONPATH="$PYTHONPATH:/yolov8_csis"

EXPOSE 7100

CMD ["python3", "app.py"]
