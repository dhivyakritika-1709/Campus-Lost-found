"""
generate_dataset.py
Synthetic Dataset Generator for Campus Lost & Found ML System.
Generates realistic pairs of Lost and Found campus reports for supervised ML training.
Includes balanced matching (label=1) and non-matching (label=0) samples across:
- Wallets, ID cards, water bottles, mobile phones, laptops, earphones,
  headphones, bags, umbrellas, watches, keys, books, calculators,
  spectacles, helmets, chargers.

Saved to: data/training_data.csv
Note: All records are synthetic demo data generated for educational CSE training.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)

ITEM_TEMPLATES = [
    {
        "category": "Electronics",
        "items": [
            ("Samsung Galaxy S22", "Black", "Black Samsung smartphone with silicone back cover", "Samsung Galaxy phone", "Black mobile found on desk with black cover"),
            ("Apple iPhone 13", "Blue", "Blue iPhone with cracked screen guard and transparent case", "Blue iPhone smartphone", "Blue Apple phone found near staircase"),
            ("HP Pavilion Laptop", "Silver", "15.6 inch silver HP laptop with Intel Core i5 sticker", "HP Laptop Silver", "Silver HP notebook laptop left in lecture hall"),
            ("Dell Inspiron Laptop", "Black", "Black Dell laptop with Linux penguin sticker on lid", "Dell Laptop", "Black Dell laptop charger and device found on bench"),
            ("Boat Rockerz Headphones", "Black", "Over-ear wireless headphones with red cushions", "Boat Bluetooth Headphones", "Black Boat wireless headset found in audio lab"),
            ("Apple AirPods Pro", "White", "AirPods Pro white charging case with initial 'D' scratched", "AirPods in white case", "Apple wireless earbuds case found on study table"),
            ("Noise ColorFit Smartwatch", "Black", "Black square dial smartwatch with magnetic charging pin", "Smartwatch with black strap", "Black digital fitness watch with touch screen"),
            ("HP 65W Laptop Charger", "Black", "Black laptop power adapter with round pin and power cord", "Laptop Charger HP", "Black HP power brick adapter found plugged in lab"),
            ("Casio FX-991EX Scientific Calculator", "Black", "Scientific calculator with slide-on protective hard cover", "Casio Scientific Calculator", "Black and white Casio calculator found in exam hall"),
            ("OnePlus Warp Charge Power Bank", "Green", "10000mAh dual output green portable power bank", "OnePlus Power Bank", "Green power bank with USB Type-C cable attached")
        ]
    },
    {
        "category": "Personal Belongings",
        "items": [
            ("Brown Leather Men's Wallet", "Brown", "Wildhorn brown leather wallet with college ID and metro card", "Brown Leather Wallet", "Gents brown wallet found near canteen cash counter"),
            ("Titan Quartz Wrist Watch", "Silver", "Analog silver metal wrist watch with date dial", "Silver wrist watch", "Metal chain watch found near library washroom"),
            ("Ray-Ban Aviator Sunglasses", "Black", "Black metallic frame sunglasses inside hard black case", "Black sunglasses in case", "Polarized sunglasses found on amphitheater steps"),
            ("Set of 3 Keys with Bike Keychain", "Silver", "Two brass keys, one bike key on Royal Enfield leather ring", "Keys on leather keychain", "Set of keys found hanging on parking gate"),
            ("Reading Spectacles with Blue Frame", "Blue", "Rectangular frame reading glasses in transparent box", "Blue Spectacles", "Spectacles with blue plastic frame found on bench")
        ]
    },
    {
        "category": "Daily Essentials",
        "items": [
            ("Milton Stainless Steel Flask 1L", "Black", "Black insulated thermal water bottle with flip lid", "Black Milton Bottle", "1-litre black steel water bottle found in classroom"),
            ("Tupperware Eco Bottle", "Blue", "750ml blue plastic water bottle with flip top cap", "Blue Tupperware bottle", "Blue reusable plastic water bottle found at gym"),
            ("Foldable Automatic Umbrella", "Navy", "Navy blue 3-fold rain umbrella with curved wooden handle", "Blue folding umbrella", "Compact navy umbrella left on corridor rack"),
            ("Vega Cruiser Motorbike Helmet", "Black", "Open face black helmet with clear visor and scratch on top", "Vega Black Helmet", "Two-wheeler black helmet found near two-wheeler parking lot"),
            ("Signoraware Insulated Lunch Bag", "Grey", "Grey fabric lunch carrier bag with two steel tiffin boxes", "Lunch bag with boxes", "Grey tiffin bag left in canteen dining area")
        ]
    },
    {
        "category": "Campus Specific",
        "items": [
            ("Student Identity Card - B.Tech CSE", "Blue", "University RFID ID card on blue ribbon lanyard", "College Student ID Card", "Blue ribbon university identity card found on counter"),
            ("Central Library Membership Card", "White", "White laminated library card with barcode and student photo", "Library Card", "Student library card found inside reference section book"),
            ("White Chemistry Lab Coat", "White", "Full sleeve cotton lab coat with embroidered name initials", "White Lab Coat", "Lab coat with pen in pocket found in chemistry lab 2")
        ]
    },
    {
        "category": "Study Materials",
        "items": [
            ("Classmate Pulse Spiral Notebook", "Yellow", "Thick 300-page notebook with DBMS and OS lecture notes", "Yellow Spiral Notebook", "Engineering lecture notes notebook found in room 204"),
            ("Data Structures & Algorithms in C++", "Blue", "Paperback textbook by D.S. Malik with highlighted notes", "C++ Algorithms Textbook", "Computer science programming textbook found in library"),
            ("Parker Vector Matte Black Fountain Pen", "Black", "Black metal body fountain pen with steel clip", "Parker Pen", "Black luxury pen found on seminar desk")
        ]
    },
    {
        "category": "Bags & Luggage",
        "items": [
            ("Skybags Casual Laptop Backpack", "Blue", "Blue and black multi-compartment college backpack", "Skybags Blue Backpack", "Blue school/college backpack found in study room"),
            ("Wildcraft Waterproof Rucksack", "Red", "Red 35L backpack with water bottle side mesh", "Red Wildcraft Bag", "Red trekking/college bag found near sports court")
        ]
    }
]

CAMPUS_LOCATIONS = [
    "Central Library 1st Floor",
    "Central Library Reading Room",
    "Main Canteen / Food Court",
    "Computer Lab 1 (Alan Turing Hall)",
    "Computer Lab 2 (Ada Lovelace Hall)",
    "Academic Block A Room 102",
    "Academic Block B Seminar Hall",
    "Mechanical Engineering Workshop",
    "Sports Complex & Gymnasium",
    "Main Auditorium Ground Floor",
    "Student Bus Stop / Pick-up Point",
    "Two-Wheeler Student Parking Bay",
    "Chemistry Laboratory 2",
    "Open Air Amphitheater",
    "Hostel Block 3 Common Room"
]


def _random_date(start_days_ago=60):
    base_date = datetime.now() - timedelta(days=start_days_ago)
    random_days = random.randint(0, start_days_ago)
    return (base_date + timedelta(days=random_days)).strftime("%Y-%m-%d")


def generate_pairs(num_pairs=1600):
    """
    Generates balanced dataset: 50% positive match pairs, 50% negative non-match pairs.
    """
    pairs = []
    half = num_pairs // 2

    # Flatten items for fast sampling
    all_items = []
    for cat_data in ITEM_TEMPLATES:
        category = cat_data["category"]
        for itm in cat_data["items"]:
            all_items.append((category, itm))

    # 1. POSITIVE MATCHES (y = 1)
    for _ in range(half):
        cat, (name_l, col_l, desc_l, name_f, desc_f) = random.choice(all_items)

        # Realistic location variation (either exact same or nearby)
        base_loc = random.choice(CAMPUS_LOCATIONS)
        if random.random() < 0.7:
            loc_l = base_loc
            loc_f = base_loc
        else:
            loc_l = base_loc
            loc_f = base_loc.split()[0] + " " + random.choice(["Corridor", "Counter", "Steps", "Lobby"])

        # Date proximity: 0 to 5 days apart
        date_obj = datetime.now() - timedelta(days=random.randint(2, 60))
        date_l = date_obj.strftime("%Y-%m-%d")
        date_f = (date_obj + timedelta(days=random.randint(0, 4))).strftime("%Y-%m-%d")

        # Color usually matches
        col_f = col_l if random.random() < 0.9 else col_l.split()[0]

        pairs.append({
            "item_name_lost": name_l,
            "category_lost": cat,
            "desc_lost": desc_l,
            "colour_lost": col_l,
            "location_lost": loc_l,
            "date_lost": date_l,
            "item_name_found": name_f,
            "category_found": cat,
            "desc_found": desc_f,
            "colour_found": col_f,
            "location_found": loc_f,
            "date_found": date_f,
            "is_match": 1
        })

    # 2. NEGATIVE MATCHES (y = 0)
    for _ in range(num_pairs - half):
        neg_type = random.random()

        if neg_type < 0.4:
            # Hard negative: Same category, different item
            cat_data = random.choice(ITEM_TEMPLATES)
            cat = cat_data["category"]
            if len(cat_data["items"]) >= 2:
                sample_2 = random.sample(cat_data["items"], 2)
                item_a, item_b = sample_2[0], sample_2[1]
            else:
                item_a = cat_data["items"][0]
                item_b = random.choice(all_items)[1]

            pairs.append({
                "item_name_lost": item_a[0],
                "category_lost": cat,
                "desc_lost": item_a[2],
                "colour_lost": item_a[1],
                "location_lost": random.choice(CAMPUS_LOCATIONS),
                "date_lost": _random_date(),
                "item_name_found": item_b[3],
                "category_found": cat,
                "desc_found": item_b[4],
                "colour_found": item_b[1],
                "location_found": random.choice(CAMPUS_LOCATIONS),
                "date_found": _random_date(),
                "is_match": 0
            })

        elif neg_type < 0.7:
            # Hard negative: Same location, completely different items
            cat_a, item_a = random.choice(all_items)
            cat_b, item_b = random.choice(all_items)
            while cat_a == cat_b:
                cat_b, item_b = random.choice(all_items)

            same_loc = random.choice(CAMPUS_LOCATIONS)
            pairs.append({
                "item_name_lost": item_a[0],
                "category_lost": cat_a,
                "desc_lost": item_a[2],
                "colour_lost": item_a[1],
                "location_lost": same_loc,
                "date_lost": _random_date(),
                "item_name_found": item_b[3],
                "category_found": cat_b,
                "desc_found": item_b[4],
                "colour_found": item_b[1],
                "location_found": same_loc,
                "date_found": _random_date(),
                "is_match": 0
            })

        else:
            # Random negative: Cross category, different dates & locations
            cat_a, item_a = random.choice(all_items)
            cat_b, item_b = random.choice(all_items)
            while cat_a == cat_b or item_a[0] == item_b[0]:
                cat_b, item_b = random.choice(all_items)

            pairs.append({
                "item_name_lost": item_a[0],
                "category_lost": cat_a,
                "desc_lost": item_a[2],
                "colour_lost": item_a[1],
                "location_lost": random.choice(CAMPUS_LOCATIONS),
                "date_lost": _random_date(start_days_ago=90),
                "item_name_found": item_b[3],
                "category_found": cat_b,
                "desc_found": item_b[4],
                "colour_found": item_b[1],
                "location_found": random.choice(CAMPUS_LOCATIONS),
                "date_found": _random_date(start_days_ago=90),
                "is_match": 0
            })

    # Shuffle dataset
    random.shuffle(pairs)
    return pd.DataFrame(pairs)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "training_data.csv")

    df = generate_pairs(num_pairs=1600)
    df.to_csv(csv_path, index=False)

    print("=" * 60)
    print("SUCCESS: Synthetic Training Dataset Generated!")
    print(f"File Path: {csv_path}")
    print(f"Total Sample Pairs: {len(df)}")
    print(f"Match Pairs (1): {sum(df['is_match'] == 1)}")
    print(f"Non-Match Pairs (0): {sum(df['is_match'] == 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
