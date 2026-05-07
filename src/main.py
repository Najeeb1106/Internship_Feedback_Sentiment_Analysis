from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os

# --- 1. Configuration & Security ---
SECRET_KEY = "your-secret-key-change-this"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="Sentintern - AI Feedback Intelligence")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# --- 2. Database (Simulated for Demo) ---
# In a real app, use SQLAlchemy with SQLite/Postgres
users_db = {} # {username: hashed_password}
feedback_history = []

# --- 3. ML Model Loading ---
MODEL_PATH = os.path.join(ROOT_DIR, "models", "finetuned_distilbert")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading ML Model from {MODEL_PATH}...")
try:
    # Use standard tokenizer from hub if local files are missing
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Fallback/Placeholder for development
    model = None

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
    if not model: return "Error", 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    
    label = "positive" if pred.item() == 1 else "negative"
    return label, float(conf.item())

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
    # In real app, verify JWT cookie here
    return templates.TemplateResponse(request, "dashboard.html", {"history": feedback_history})

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
        "history": feedback_history
    })

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("access_token")
    return response
