import joblib
import numpy as np
import pandas as pd
import io
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# входной массив флотов (паказаний приборов) > переделали на строку из-за индекса дататайма, обычно индексы не используются
class Model(BaseModel):
    X: list[str]

app = FastAPI() 

# инициализация модели
loaded_model = joblib.load("model_baseline.pkl") # загружаем модель из бинарника в корне

# инструкция для FastAPI c эндпоинтом predict
@app.post("/predict")

def predict_model(model: Model):
    #print(model.X)
    string_data = """datetime,Accelerometer1RMS,Accelerometer2RMS,Current,Pressure,Temperature,Thermocouple,Voltage,Volume Flow RateRMS
    """ + '\r\n'.join(model.X)
    ##df = pd.DataFrame(io.StringIO(string_data), sep = ';', index_col = 'datetime') # преабразуем входной поток в массив флотов
    df = pd.read_csv(io.StringIO(string_data), sep = ',', index_col = 'datetime') # преабразуем входной поток в массив флотов
    ##samples_to_predict = np.array(model.X).reshape(1, -1) # преабразуем переданные значения в одномерный массив
    ##result = loaded_model.predict(saple_to_predict) # генерация предсказания
    result = loaded_model.predict(df) # генерация предсказания
    return {"result:": ','.join(map(str, result))}

# для отладки
def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()







