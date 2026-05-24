# Mirror: Behavioral Persona Recommendation & Synthesis Engine

Mirror is an advanced machine learning framework that models individual human behavior to generate personalized reviews and product recommendations. By constructing a **Latent Persona Vector (Pu)** using psychometric and linguistic data, Mirror bridges the gap between generic AI outputs and authentic human voice.

---

##  Key Features

*   **Multi-Modal Profiling**: Acquisition of user personality via three layers:
    1.  **Scenario Tensions**: Behavioral choices mapped to OCEAN (Big Five) traits.
    2.  **Linguistic Fingerprinting**: LIWC-based analysis of writing samples (Type-Token Ratio, Affective Ratios).
    3.  **Rating Calibration**: Quantifying personal rating bias (harsh vs. generous).
*   **Archetype Modeling**: Gaussian Mixture Model (GMM) with **8 Latent Archetypes** (Critic, Enthusiast, Loyalist, etc.) allowing for "Blended Personas."
*   **Predictive Scoring**: XGBoost-powered ordinal regressor trained on 477,000+ interactions (**MAE: 0.71**).
*   **Behavioral Synthesis**: A "Metacognitive" Two-Call LLM Chain (Reasoning + Generation) using **Google Gemini** to mirror the user's authentic voice.
*   **Real-World Data Integration**: Live scrapers for **Google Play Store**, **Amazon**, and **Jumia Nigeria**.

---

##  System Architecture

*   **Backend**: Django 5.2 & Django REST Framework.
*   **ML Stack**: Scikit-Learn, XGBoost, Pandas, Numpy 1.26.
*   **AI Engine**: Google Gemini 1.5/2.0 Flash.
*   **Evaluation**: BERTScore (Semantic Similarity) & ROUGE-L (Structural fidelity).
*   **Infrastructure**: Docker, PostgreSQL, Django Q2 (Background Workers), WhiteNoise.

---

##  Getting Started

### 1. Prerequisites
- Python 3.11+
- [Google Gemini API Key](https://aistudio.google.com/)
- [TMDB API Key](https://www.themoviedb.org/documentation/api) (for movie metadata)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/ImperiumBuild/mirror.git
cd Mirror

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
DEBUG=True
DJANGO_SECRET_KEY=your_secret_key
GEMINI_API_KEY=your_primary_key
GEMINI_API_KEY_2=your_second_key (Optional for rotation)
TMDB_API_KEY=your_tmdb_key
DB_NAME=mirror_db
DB_USER=postgres
DB_PASSWORD=your_pass
DB_HOST=localhost
DB_PORT=5432
```

### 4. Database Setup
```bash
cd mirror_app
python manage.py migrate
python manage.py collectstatic --noinput
```

---

##  Running the Application

### Local Development
1. **Start the Django Server**:
   ```bash
   python manage.py runserver
   ```
2. **Start the Background Worker** (Required for persona updates):
   ```bash
   python manage.py qcluster
   ```

### Docker Deployment
```bash
docker-compose up --build
```

---

## 🧪 Evaluation Pipeline (User Study)

Mirror includes a robust pipeline to test accuracy against real-world data (e.g., your Google Forms CSV).

1.  **Generate Reviews**:
    Uses batch processing (14 users/call) to generate AI reviews for the 70-user dataset.
    ```bash
    python evaluate_google_forms.py
    ```
2.  **Calculate Metrics**:
    Calculates BERTScore and ROUGE-L between AI and Human reviews.
    ```bash
    python calculate_metrics.py
    ```
    *Note: If you experience NumPy crashes, the script automatically uses a TF-IDF Semantic Fallback.*

---

## Empirical Results
Based on a study of 70 unique users:
- **BERTScore**: 0.7235 (High semantic alignment)
- **Rating MAE**: 0.71 stars (High predictive precision)
- **ROUGE-L**: 0.0914 (Captured voice structural patterns)

---

## 📄 Documentation
For a deep dive into the math, archetypes, and model specifications, see:
- `MIRROR_TECHNICAL_PAPER.txt`: Comprehensive technical white paper.
- `CORE_ALGORITHM.md`: Architectural overview of the ML layers.

---
© 2026 Mirror AI Project.
