# mirror_app/management/commands/load_data.py
from django.core.management.base import BaseCommand
import json, csv
from recommendations.models import ReviewBank, UserReviewProfile

class Command(BaseCommand):
    help = "Load review bank and user profiles into database"

    def handle(self, *args, **kwargs):
        # load review bank
        self.stdout.write("Loading review bank...")
        for path in ["data/processed/review_bank.json",
                     "data/processed/review_bank_apps.json"]:
            try:
                with open(path) as f:
                    bank = json.load(f)
                objs = [ReviewBank(
                    category     = r.get("category", ""),
                    sub_category = r.get("sub_category", ""),
                    item_id      = r.get("item_id", ""),
                    title        = r.get("title", ""),
                    text         = r.get("text", ""),
                    rating       = r.get("rating", 3),
                    ocean_o      = r.get("ocean_O", 0.5),
                    ocean_c      = r.get("ocean_C", 0.5),
                    ocean_e      = r.get("ocean_E", 0.5),
                    ocean_a      = r.get("ocean_A", 0.5),
                    ocean_n      = r.get("ocean_N", 0.5),
                    generosity_score  = r.get("generosity_score", 0.5),
                    avg_review_length = r.get("avg_review_length", 0.0),
                    source       = r.get("source", "amazon"),
                ) for r in bank]
                ReviewBank.objects.bulk_create(objs, ignore_conflicts=True)
                self.stdout.write(f"  ✓ {path}: {len(objs)} reviews loaded")
            except FileNotFoundError:
                self.stdout.write(f"  WARNING: {path} not found")

        # load user profiles
        self.stdout.write("Loading user profiles...")
        try:
            with open("data/processed/user_archetypes.csv") as f:
                reader = csv.DictReader(f)
                objs = [UserReviewProfile(
                    user_id              = row["user_id"],
                    source               = row.get("source", "amazon"),
                    dominant_archetype   = row.get("dominant_archetype", ""),
                    secondary_archetype  = row.get("secondary_archetype", ""),
                    archetype_confidence = float(row.get("archetype_confidence", 0)),
                    ocean_o              = float(row.get("ocean_O", 0.5)),
                    ocean_c              = float(row.get("ocean_C", 0.5)),
                    ocean_e              = float(row.get("ocean_E", 0.5)),
                    ocean_a              = float(row.get("ocean_A", 0.5)),
                    ocean_n              = float(row.get("ocean_N", 0.5)),
                    avg_rating           = float(row.get("avg_rating", 0)),
                    generosity_score     = float(row.get("generosity_score", 0.5)),
                    avg_review_length    = float(row.get("avg_review_length", 0)),
                ) for row in reader]
                UserReviewProfile.objects.bulk_create(
                    objs, ignore_conflicts=True, batch_size=1000)
                self.stdout.write(f"  ✓ {len(objs)} user profiles loaded")
        except FileNotFoundError:
            self.stdout.write("  WARNING: user_archetypes.csv not found")

        self.stdout.write(self.style.SUCCESS("Data load complete."))