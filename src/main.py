from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os

# --- 1. Configuration & Security ---
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
MODEL_PATH = os.getenv("MODEL_PATH", "najeeb786/sentintern-ai")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check model status on startup
    global model_online
    try:
        # Use client to check status
        print(f"Verifying model {MODEL_PATH} status...")
        # A simple call to check if it's there
        # We don't use the result, just check if it raises an error
        try:
            client.text_classification("Startup test")
            print("Successfully connected to Hugging Face Inference API!")
            model_online = True
        except Exception as e:
            err_msg = str(e).lower()
            if "loading" in err_msg or "503" in err_msg:
                print("Model is currently loading on Hugging Face. It will be ready soon.")
                model_online = True # It's online, just warming up
            else:
                print(f"Startup Model Warning: {e}")
                # Check if we have a local model as fallback
                if os.path.exists(os.path.join(ROOT_DIR, "models", "finetuned_distilbert")):
                    print("Found local model. Using local fallback for development.")
                    model_online = True
                else:
                    model_online = False
    except Exception as e:
        print(f"Startup Connection Failed: {e}")
        model_online = False
    yield

app = FastAPI(title="Sentintern - AI Feedback Intelligence", lifespan=lifespan)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"GLOBAL ERROR: {exc}")
    return HTMLResponse(content=f"Internal Server Error: {exc}", status_code=500)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# --- 2. Database (Simulated for Demo) ---
# In a real app, use SQLAlchemy with SQLite/Postgres
# WARNING: In-memory storage will reset on Render Free Tier restarts
users_db = {} # {username: hashed_password}
feedback_history = []

# --- 3. ML Model Loading (Using HF Inference API) ---
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN", "")
client = InferenceClient(model=MODEL_PATH, token=HF_TOKEN)
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_PATH}"
model_online = True # We'll verify during lifespan

def query_hf_api(payload):
    # This is kept for backward compatibility if needed, 
    # but we'll use client directly in predict_sentiment
    import requests
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return {"error": f"Status {response.status_code}: {response.text}"}
    return response.json()

print(f"Connected to HF Inference API for {MODEL_PATH}")

# --- 4. Helper Functions ---
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def predict_sentiment(text):
    global model_online
    try:
        # 1. Try HF Inference API first
        try:
            results = client.text_classification(text)
            if results:
                # Find the result with highest score
                best = max(results, key=lambda x: x['score'])
                
                label = best['label'].lower()
                # Map labels if they are LABEL_0/LABEL_1
                if label == "label_1": label = "positive"
                elif label == "label_0": label = "negative"
                
                return label, float(best['score'])
        except Exception as hf_err:
            err_msg = str(hf_err).lower()
            if "loading" in err_msg or "503" in err_msg:
                print("HF API: Model is still loading...")
                return "Loading...", 0.0
            print(f"HF API Error: {hf_err}")
            
        # 2. Fallback to Local Model if API fails and we are local
        local_path = os.path.join(ROOT_DIR, "models", "finetuned_distilbert")
        if os.path.exists(local_path):
            print("Using local model fallback...")
            # We don't want to load transformers here every time, but for a quick fix:
            from transformers import pipeline
            local_pipe = pipeline("text-classification", model=local_path, tokenizer=local_path)
            res = local_pipe(text)[0]
            label = res['label'].lower()
            if label == "label_1": label = "positive"
            elif label == "label_0": label = "negative"
            return label, float(res['score'])
            
    except Exception as e:
        print(f"Prediction Error: {e}")
        model_online = False
    
    return "Error", 0.0

# --- 5. Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "home.html")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    user_pw = users_db.get(username)
    if not user_pw or not verify_password(password, user_pw):
        return RedirectResponse(url="/login?error=Invalid credentials", status_code=status.HTTP_303_SEE_OTHER)
    
    token = create_access_token({"sub": username})
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="access_token", value=f"Bearer {token}", httponly=True)
    return response

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse(request, "signup.html")

@app.post("/signup")
async def signup(username: str = Form(...), password: str = Form(...)):
    if username in users_db:
        return RedirectResponse(url="/signup?error=User exists", status_code=status.HTTP_303_SEE_OTHER)
    users_db[username] = get_password_hash(password)
    return RedirectResponse(url="/login?msg=Account created", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {
        "history": feedback_history,
        "model_online": model_online
    })

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(request, "about.html")

@app.get("/how-it-works", response_class=HTMLResponse)
async def how_it_works(request: Request):
    return templates.TemplateResponse(request, "how_it_works.html")

@app.post("/analyze")
async def analyze(request: Request, feedback: str = Form(...)):
    label, confidence = predict_sentiment(feedback)
    result = {
        "text": feedback,
        "sentiment": label,
        "confidence": f"{confidence:.1%}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    feedback_history.insert(0, result)
    return templates.TemplateResponse(request, "dashboard.html", {
        "result": result, 
        "history": feedback_history,
        "model_online": model_online
    })

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("access_token")
    return response
