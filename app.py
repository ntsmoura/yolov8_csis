import datetime
import logging
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import uvicorn
from dynaconf import Dynaconf

from fastapi import FastAPI, Form, UploadFile, Response
from starlette.responses import JSONResponse
from ultralytics import YOLO
import cv2

import firebase_admin
from firebase_admin import credentials, firestore_async
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

app = FastAPI()

settings = Dynaconf(
    settings_files=[".settings.local.toml"],
    root_path=Path(__file__).parent,
    merge_enabled=True,
)

model = YOLO("models/best_public_safety_30.pt")

cred = credentials.Certificate(settings.CRED.json_name)

firebase_admin.initialize_app(cred)

db = firestore_async.client()

predictions = db.collection("predictions")

if not os.path.exists("./received_images"):
    os.makedirs("./received_images")


@app.get("/health_check")
async def health_check():
    result = model.predict(source="health_check_img.jpg", device="cpu")
    assert int(result[0].boxes.cls[0]) == 8
    assert int(result[0].boxes.cls[1]) == 2
    return JSONResponse(content="API e Modelo funcionando como esperado...")


@app.get("/metrics")
async def metrics():
    correct_preds = (await predictions.where(filter=FieldFilter("feedback", "==", "CORRECT")).count().get())[0][0].value
    wrong_preds = (await predictions.where(filter=FieldFilter("feedback", "==", "WRONG")).count().get())[0][0].value
    pending_feedbacks = (await predictions.where(filter=FieldFilter("feedback", "==", "PENDING")).count().get())[0][
        0
    ].value
    division = None
    if correct_preds or wrong_preds:
        division = correct_preds / (correct_preds + wrong_preds)

    return JSONResponse(
        status_code=200,
        content={
            "predicoes_corretas": correct_preds,
            "predicoes_erradas": wrong_preds,
            "predicos_sem_feedback": pending_feedbacks,
            "razao": division,
        },
    )


@app.post("/{predict_id}/report")
async def report(status: str, predict_id: str):
    pred_ref = predictions.document(predict_id)
    if not (await pred_ref.get()).exists:
        return JSONResponse(
            status_code=404,
            content={"message": "Predição não encontrada."},
        )
    match status:
        case "correct":
            os.remove(f"./received_images/{predict_id}.png")
            await pred_ref.update({"feedback": "CORRECT", "last_update": datetime.datetime.now()})
        case "wrong":
            await pred_ref.update({"feedback": "WRONG", "last_update": datetime.datetime.now()})
        case _:
            return JSONResponse(
                status_code=400,
                content={"message": "Tipo de report não suportado"},
            )
    return JSONResponse(status_code=200, content={"message": "Feedback recebido."})


@app.post("/predict/{return_type}")
async def predict(image: Annotated[UploadFile, Form()], return_type: str):

    _, prediction_ref = await predictions.add(
        {"last_update": datetime.datetime.now(), "status": "RECEIVED", "feedback": "PENDING"}
    )
    try:
        img_bytes = await image.read()
        raw_img = np.asarray(bytearray(img_bytes), dtype="uint8")
        raw_img = cv2.imdecode(raw_img, cv2.IMREAD_COLOR)
        cv2.imwrite(f"./received_images/{prediction_ref.id}.png", raw_img)

        result = model.predict(source=raw_img, device="cpu")
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
            await prediction_ref.update({"status": "OK"})
            return JSONResponse(content=results, headers={"doc_id": prediction_ref.id})
        elif return_type == "img":
            result_img = cv2.imencode(".png", result[0].plot())
            await prediction_ref.update({"status": "OK"})
            return Response(
                content=result_img[1].tobytes(), media_type="image/png", headers={"doc_id": prediction_ref.id}
            )
    except Exception as error:
        logging.error(error)
        await prediction_ref.update({"status": "ERROR"})
        return JSONResponse(status_code=500, content={"message": "Erro desconhecido"})
    else:
        await prediction_ref.update({"status": "ERROR"})
        return JSONResponse(
            status_code=400,
            content={"message": "Tipo de retorno não suportado"},
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7100)
