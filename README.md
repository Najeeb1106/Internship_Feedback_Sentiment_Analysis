# 🧠 Sentintern AI: Advanced Feedback Intelligence

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)

Sentintern AI is a high-performance, full-stack sentiment analysis platform that transforms raw qualitative feedback into quantitative business intelligence. Leveraging a fine-tuned **DistilBERT** transformer model, it provides sub-second inference with industry-leading accuracy.

![Dashboard Preview](src/static/dashboard_preview.png)

## ✨ Key Features

- **🎯 Precision Inference**: Fine-tuned DistilBERT model achieving ~100% accuracy on domain-specific feedback.
- **📊 Real-time Analytics**: Interactive dashboard with Chart.js for sentiment distribution and confidence trends.
- **🛡️ Secure Architecture**: JWT-based authentication with encrypted session management.
- **⚡ High Performance**: Built with FastAPI for asynchronous request handling and sub-second model response.
- **✨ Glassmorphism UI**: Modern, premium user interface designed with sleek dark-mode aesthetics and fluid animations.

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.10+)
- **Machine Learning**: PyTorch, Transformers (DistilBERT)
- **Frontend**: Jinja2 Templates, Vanilla CSS (Modern CSS Hub), Lucide Icons
- **Visualization**: Chart.js
- **Authentication**: Python-Jose (JWT), Passlib (Bcrypt)

## 📁 Repository Structure

```text
Sentintern_AI/
├── src/                # FastAPI Application
│   ├── static/         # CSS, Images, JS
│   ├── templates/      # Jinja2 HTML Templates
│   └── main.py         # Entry Point
├── models/             # ML Model Weights & Config
├── notebooks/          # Training & Research Notebooks
├── data/               # Datasets (CSV)
├── docs/               # Technical Documentation
├── .env.example        # Environment Template
└── requirements.txt    # Dependency Manifest
```

## 🚀 Local Setup

### 1. Clone & Install
```bash
git clone https://github.com/Najeeb1106/Internship_Feedback_Sentiment_Analysis.git
cd Internship_Feedback_Sentiment_Analysis
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your SECRET_KEY
```

### 3. Initialize Model
Ensure your fine-tuned model files (`pytorch_model.bin`, `config.json`) are located in:
`models/finetuned_distilbert/`

### 4. Run Development Server
```bash
cd src
uvicorn main:app --reload
```
Visit [http://localhost:8000](http://localhost:8000)

## 📐 Architecture

For a deep dive into the system design, model fine-tuning process, and API specifications, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 📈 Future Roadmap

- [ ] **Persistent Database**: Transition from in-memory history to PostgreSQL.
- [ ] **Bulk Upload**: Support for CSV/Excel file uploads for batch processing.
- [ ] **Multi-language Support**: Expand sentiment analysis to non-English feedback.
- [ ] **Export Reports**: Generate PDF/CSV summaries of feedback trends.

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Built for excellence in Feedback Intelligence.*
