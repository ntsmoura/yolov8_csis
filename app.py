from typing import Annotated

import numpy as np
import uvicorn

from fastapi import FastAPI, Form, UploadFile, Response
from starlette.responses import JSONResponse
from ultralytics import YOLO
import cv2

app = FastAPI()

model = YOLO("best_public_safety_10.pt")


@app.get("/health_check")
async def health_check():
    result = model.predict(source="health_check_img.jpg", device="cpu")
    assert int(result[0].boxes.cls[0]) == 8
    assert int(result[0].boxes.cls[1]) == 2
    return JSONResponse(content="API e Modelo funcionando como esperado...")


@app.post("/predict/{return_type}")
async def predict(image: Annotated[UploadFile, Form()], return_type: str):
    img_bytes = await image.read()
    raw_img = np.asarray(bytearray(img_bytes), dtype="uint8")
    raw_img = cv2.imdecode(raw_img, cv2.IMREAD_COLOR)

    result = model.predict(source=raw_img, device="cpu")
    cv2.imwrite("result.png", result[0].plot())
    if return_type == "json":
        names = result[0].names
        classes = result[0].boxes.cls
        confs = result[0].boxes.conf
        results = {}
        if classes is not None:
            for i in range(len(classes)):
                label = names[int(classes[i])]
                if label not in results:
                    results[label] = [float(confs[i])]
                else:
                    results[label].append(float(confs[i]))
        return JSONResponse(content=results)
    elif return_type == "img":
        result_img = cv2.imencode(".png", result[0].plot())
        return Response(content=result_img[1].tobytes(), media_type="image/png")
    else:
        return JSONResponse(
            status_code=400,
            content={"message": "Tipo de retorno não suportado"},
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7100)
