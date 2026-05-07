# Technical Architecture: Sentintern AI

## 1. System Overview
Sentintern AI follows a classic client-server architecture with a heavy focus on the Machine Learning inference pipeline. The system is designed to handle high-frequency sentiment analysis requests with minimal latency.

## 2. ML Inference Pipeline
The core intelligence layer utilizes **DistilBERT**, a smaller, faster, cheaper, and lighter version of BERT.

- **Preprocessing**: Input text is cleaned (lowercase, URL removal, special character stripping) and tokenized using `DistilBertTokenizerFast`.
- **Model**: `DistilBertForSequenceClassification` fine-tuned on 4,000 unique feedback entries.
- **Output**: The model returns raw logits, which are converted to probabilities using a Softmax layer, providing both the **Sentiment Label** and a **Confidence Score**.

## 3. Backend Architecture (FastAPI)
The backend is built for speed and modularity:
- **Router Logic**: Clean separation of Auth, Analytics, and Page routes.
- **Middleware**: Handles secure session cookies and JWT verification.
- **State Management**: (Current) In-memory list for session history; (Future) SQL-based persistence.

## 4. Frontend & UI/UX
- **Design System**: A custom glassmorphism system using CSS variables for consistent theming.
- **Dynamic Charting**: Real-time updates to the dashboard via Chart.js, visualizing history data injected from the backend.
- **Iconography**: Lucide icons integrated for a modern, lightweight SVG experience.

## 5. Security Model
- **Authentication**: Bcrypt-hashed passwords stored in the user database.
- **Authorization**: Stateless JWT tokens stored in `HttpOnly` cookies to prevent XSS-based token theft.
- **Environment**: All sensitive configurations are managed via `.env` files.

## 6. Training Process
> [!NOTE]
> The model weights generated from this process (approx. 260MB) are ignored by Git. When deploying or running locally, ensure the `.safetensors` or `.bin` files are manually placed in the `models/` directory.

The model was trained using the following parameters:
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 5e-5
- **Optimizer**: AdamW
- **Platform**: Google Colab (T4 GPU)
