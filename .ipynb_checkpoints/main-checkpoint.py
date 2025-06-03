from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # Hiçbir prediction göndermiyoruz!
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/", response_class=HTMLResponse)
async def predict_news(request: Request, haber: str = Form(...)):
    vect = vectorizer.transform([haber])
    sonuc = model.predict(vect)[0]
    prediction = "Gerçek" if sonuc == 1 else "Sahte"
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})