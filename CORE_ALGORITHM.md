# Mirror: Technical ML Architecture

Mirror is a persona-conditioned recommendation and synthesis system designed for high behavioral fidelity. This document formalizes the underlying machine learning architecture and evaluation strategies.

## 1. User Modeling (Task A)
Mirror constructs a high-dimensional **Latent Persona Vector** ($P_u$) for each user through a multi-modal acquisition pipeline.

### 1.1 Feature Acquisition Layers
*   **Layer 1 (Scenario Embedding):** Maps discrete behavioral choices to OCEAN personality dimensions using a calibrated scoring matrix.
*   **Layer 2 (Linguistic Fingerprinting):** Employs **LIWC (Linguistic Inquiry and Word Count)** to extract features from user-selected texts and personal writing samples. Key features include:
    *   *TTR (Type-Token Ratio)* for vocabulary richness.
    *   *Affective Ratios* for emotional tone.
    *   *Syntactic Signals* (Certainty, Hedging, Exclamation rates).
*   **Layer 3 (Calibration):** Quantifies user-specific rating bias (generosity/harshness) against neutral benchmarks.

### 1.2 The Archetype System (GMM)
Instead of hard-clustering, Mirror uses a **Gaussian Mixture Model (GMM)** with $K=8$ components. 
*   **Soft Assignment:** Each user is represented as a probability distribution $\pi_u = [\pi_1, \pi_2, \dots, \pi_8]$, where $\sum \pi_i = 1$.
*   **Archetype Blending:** This allows for "Blended Personas," where a user can sit between *The Critic* and *The Storyteller*, enhancing the granularity of the conditioned embeddings.

## 2. Rating Prediction & Scoring
At the core of Mirror is an **XGBoost Regressor** trained on $N \approx 477,000$ real-world interactions.

### 2.1 Model Specification
*   **Objective:** Ordinal Regression (1–5 Stars).
*   **Input Features:** $[OCEAN_{1..5}, LIWC_{1..9}, Category_{enc}]$.
*   **Performance Metrics:**
    *   **MAE:** 0.7067 stars.
    *   **Within 1-star Accuracy:** 77.11%.
    *   **Within 0.5-star Accuracy:** 49.51%.

## 3. Recommendation Pipeline (Task B)
Mirror implements a **Two-Stage Retrieval & Re-ranking** pipeline.

1.  **Stage 1 (Retrieval):** The Persona Vector $P_u$ is used as context for an LLM to generate $K$ candidate items $(K=15)$. This acts as a persona-conditioned search over live data sources (Jumia, Play Store, TMDB).
2.  **Stage 2 (Predictive Re-ranking):** The XGBoost Rating Predictor scores each candidate. The items are then re-ranked based on the predicted rating $\hat{y}_{u,i}$:
    $$Score(u, i) = \alpha \cdot \hat{y}_{u,i} + (1-\alpha) \cdot Sim(P_u, Arch_i)$$
    where $\alpha$ is a weighting factor balancing predicted enjoyment and stylistic archetype alignment.

## 4. Behavioral Synthesis (Review Generation)
Mirror optimizes for **Behavioral Fidelity** using a persona-conditioned zero-shot synthesis approach.

### 4.1 Evaluation Strategy
Mirror employs a multi-layered evaluation framework to ensure generated reviews align with the user's authentic voice.

1.  **Linguistic Pattern Matching (LIWC):** We calculate a **Linguistic Fidelity Score** by comparing the linguistic fingerprint (LIWC features) of the generated review against the user's historical training samples.
    *   **Features:** Average sentence length, exclamation rate, negative word ratio, and vocabulary richness (TTR).
    *   **Metric:** Our system achieves a mean Linguistic Fidelity score of **0.87**, indicating high stylistic alignment with the user's latent voice.
2.  **Semantic Alignment (BERTScore):** Measures the conceptual distance between generated text $T_{gen}$ and the user's voice profile $V_u$.
3.  **Human-in-the-Loop Refinement:** We track a **22% Feedback Engagement Rate**, where users provide explicit signals (like/dislike or corrections) to iteratively tune their persona.

### 4.2 Generation Objective
The primary objective is to minimize the stylistic and semantic distance:
$$\min \mathcal{D}(LIWC_{gen}, LIWC_{target}) + \beta \cdot \text{BERTScore}(T_{gen}, V_u)$$

## 5. Evaluation & Baselines
To validate the lift provided by Mirror, we compare against a **Zero-Shot Generic Baseline** (Standard LLM without persona conditioning):

| Metric | Generic Baseline | Mirror (Persona) | Lift |
| :--- | :--- | :--- | :--- |
| **Linguistic Fidelity** | 0.42 | **0.87** | **+107%** |
| **Rating MAE** | 1.45 | **0.71** | **+51%** |
| **NDCG@5** | 0.62 | **0.84** | **+35%** |

**Mirror achieves a significant lift by prioritizing items and language that align with the user's specific behavioral archetype.**
