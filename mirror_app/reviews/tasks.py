"""
reviews/tasks.py
Background job — updates user profile after feedback.
Called by Django Q2.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from users.models import User, TasteProfile
from core.ocean.profile_builder import update_profile_from_feedback


def update_profile_task(
    user_id:          int,
    feedback_score:   str,
    generated_review: str,
    edited_review:    str,
    category:         str,
):
    try:
        user = User.objects.get(id=user_id)
        tp   = TasteProfile.objects.get(user=user)

        updated_profile = update_profile_from_feedback(
            existing_profile = tp.profile_json,
            feedback_score   = feedback_score,
            generated_review = generated_review,
            edited_review    = edited_review or None,
            category         = category,
        )

        ocean = updated_profile.get("ocean", {})

        tp.profile_json        = updated_profile
        tp.ocean_o             = ocean.get("O", tp.ocean_o)
        tp.ocean_c             = ocean.get("C", tp.ocean_c)
        tp.ocean_e             = ocean.get("E", tp.ocean_e)
        tp.ocean_a             = ocean.get("A", tp.ocean_a)
        tp.ocean_n             = ocean.get("N", tp.ocean_n)
        tp.dominant_archetype  = updated_profile.get("dominant_archetype", tp.dominant_archetype)
        tp.secondary_archetype = updated_profile.get("secondary_archetype", tp.secondary_archetype)
        tp.feedback_count      = updated_profile.get("feedback_count", tp.feedback_count)
        tp.profile_summary     = updated_profile.get("profile_summary", tp.profile_summary)
        tp.save()

    except Exception as e:
        print(f"[update_profile_task] Error for user {user_id}: {e}")