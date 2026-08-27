from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os
from app.services.cache_retry_scheduler import start_cache_retry_scheduler
load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
def on_startup():
    start_cache_retry_scheduler()
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        user_message = req.message.strip()
        if not user_message:
            return JSONResponse({"answer": "请输入问题"}, status_code=400)

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个简洁、专业的智能助手。"},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )

        answer = completion.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        return JSONResponse({"answer": f"调用失败：{str(e)}"}, status_code=500)