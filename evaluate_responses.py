
import pandas as pd
import json
import os
from rouge_score import rouge_scorer
from bert_score import score as calculate_bert_score
from core.ocean.profile_builder import build_profile
from core.review_engine import ReviewEngine

# --- CONFIGURATION ---
INPUT_CSV = "responses.csv"  # The file from Google Forms
OUTPUT_CSV = "evaluation_results.csv"

def process_evaluations():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Please place your Google Forms export here.")
        return

    df = pd.read_csv(INPUT_CSV)
    results = []
    
    # Initialize Engines
    engine = ReviewEngine(provider="gemini")
    r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    print(f"--- Processing {len(df)} responses ---")

    for index, row in df.iterrows():
        print(f"Processing row {index + 1}...")

        # 1. Map Layer 1 Answers (Example mapping)
        # Note: Match these keys to your CSV headers
        layer1 = {
            "q_damage": row.get("q_damage_answer", "b"),
            "q_app_bug": row.get("q_app_bug_answer", "b"),
            "q_youtube_movie": row.get("q_youtube_answer", "a"),
            "q_oraimo_delivery": row.get("q_oraimo_answer", "b"),
            "q_doctor_vibe": row.get("q_doctor_answer", "c"),
        }

        # 2. Map Layer 2 (Pairwise / Own Writing)
        # We assume the CSV has columns for chosen text and tag
        pairwise = [
            {"chosen_text": row.get("scenario_1_text"), "tag": row.get("scenario_1_tag", "own_writing")},
            {"chosen_text": row.get("scenario_2_text"), "tag": row.get("scenario_2_tag", "own_writing")},
        ]

        # 3. Map Layer 3 (Calibration)
        calibration = {
            "c_delivery": int(row.get("cal_delivery", 3)),
            "c_movie": int(row.get("cal_movie", 3)),
        }

        # 4. Build Persona
        try:
            profile = build_profile(
                layer1_answers=layer1,
                pairwise_picks=pairwise,
                calibration_answers=calibration,
                user_reviews=[row.get("user_review", "")]
            )
        except Exception as e:
            print(f"Profile build failed for row {index}: {e}")
            continue

        # 5. Generate Mirror AI Review
        item_name = row.get("item_name", "Unknown Item")
        category = row.get("category", "apps").lower()
        
        gen_result = engine.generate_review(
            profile=profile,
            product_name=item_name,
            category=category
        )
        ai_review = gen_result["review"]
        user_review = row.get("user_review", "")

        # 6. Calculate Metrics
        # ROUGE-L
        rouge_scores = r_scorer.score(user_review, ai_review)
        rouge_l = rouge_scores['rougeL'].fmeasure

        # BERTScore
        P, R, F1 = calculate_bert_score([ai_review], [user_review], lang="en", model_type="distilbert-base-uncased", verbose=False)
        bert_f1 = float(F1[0])

        # Rating Error
        rating_error = abs(gen_result["predicted_rating"] - int(row.get("user_rating", 3)))

        # 7. Collect Results
        results.append({
            "User_ID": index + 1,
            "Archetype": profile["dominant_archetype"],
            "Item": item_name,
            "User_Review": user_review,
            "AI_Review": ai_review,
            "ROUGE_L": round(rouge_l, 4),
            "BERTScore": round(bert_f1, 4),
            "Rating_Error": rating_error,
            "Traits": str(profile["ocean"])
        })

    # Export to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"--- Evaluation Complete! Results saved to {OUTPUT_CSV} ---")

if __name__ == "__main__":
    process_evaluations()
