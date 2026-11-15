#!/usr/bin/env python3
"""Prepare M3Kang for the Arduino UNO Q, on your laptop.

M3Kang stores each problem as text plus an image. The text has the question
but not the five answer options: those only appear in the image. This script:

1. downloads the parquet shards you ask for from Hugging Face,
2. keeps the languages you ask for,
3. reads the options A to E from each image with OCR (RapidOCR, offline),
4. splits the problems into train and validation by problem ID, so a
   problem and its translations never land on both sides,
5. writes small JSONL files (no images) that you copy to the board.

Usage:
    python3 laptop/prepare_m3kang.py                      # English and Catalan
    python3 laptop/prepare_m3kang.py --languages eng,cat,esp
    python3 laptop/prepare_m3kang.py --val-years 2024     # hold out a whole year
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import Counter

DATASET = "qualcomm/M3Kang"
SHARD = "data/{split}-{index:05d}-of-{total:05d}.parquet"
SHARDS = {"high_quality": 4, "standard": 10}
LETTERS = "ABCDE"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="high_quality", choices=sorted(SHARDS))
    p.add_argument("--shards", default="0",
                   help="comma-separated shard numbers, or 'all'. Shard 0 of high_quality has "
                        "eng, cat, fra, esp, ind, deu, swh, vie, tur, zho, kor, arb, jpn, lit")
    p.add_argument("--parquet", nargs="*", help="use local parquet files instead of downloading")
    p.add_argument("--languages", default="eng,cat", help="M3Kang language codes, e.g. eng,cat,esp")
    p.add_argument("--val-fraction", type=float, default=0.15, help="share of problem IDs held out")
    p.add_argument("--val-years", default="", help="hold out these years instead, e.g. 2023,2024")
    p.add_argument("--out", default="data", help="output folder")
    p.add_argument("--no-ocr", action="store_true", help="skip OCR; options will be empty")
    p.add_argument("--save-images", action="store_true", help="also save problem images as PNG")
    p.add_argument("--limit", type=int, default=0, help="only process the first N rows (for testing)")
    return p.parse_args()


def shard_paths(args):
    if args.parquet:
        return args.parquet
    from huggingface_hub import hf_hub_download
    total = SHARDS[args.split]
    indexes = range(total) if args.shards == "all" else [int(s) for s in args.shards.split(",")]
    paths = []
    for i in indexes:
        name = SHARD.format(split=args.split, index=i, total=total)
        print(f"Downloading {name} (cached after the first time)...", flush=True)
        paths.append(hf_hub_download(DATASET, name, repo_type="dataset"))
    return paths


# --- Reading the options from the image -------------------------------------
#
# OCR returns text boxes. The options are the last A..E sequence in reading
# order, usually on one row at the bottom. OCR noise we handle, seen on real
# M3Kang images: "A)21" (no space), "E16" (lost parenthesis), "C" and "13" in
# separate boxes, ")768351" (parenthesis attached to the value), and circled
# letters like "A) (A)" where the circle is dropped or misread.

MARKER = re.compile(r"^\s*([A-E])\s*([\)）\]:.]?)\s*(.*)$")


def preprocess_image(image_bytes, crop_bottom=False):
    """Upscale 2x and add white padding so small digits/circled letters are detected cleanly."""
    from PIL import Image, ImageOps
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if crop_bottom:
        # Focus on the bottom 45% of the image to avoid diagram labels at the top/right
        img = img.crop((0, int(h * 0.55), w, h))
        w, h = img.size
    img = img.resize((w * 2, h * 2), Image.Resampling.BICUBIC)
    img = ImageOps.expand(img, border=32, fill="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def reading_order(boxes):
    """Sort OCR boxes into rows (top to bottom), then left to right."""
    items = []
    for box, text, _score in boxes:
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        items.append({
            "x": min(xs),
            "x_max": max(xs),
            "y": (min(ys) + max(ys)) / 2,
            "h": max(ys) - min(ys),
            "text": text,
        })
    items.sort(key=lambda it: it["y"])
    rows, current = [], []
    for it in items:
        if current:
            row_y = sum(x["y"] for x in current) / len(current)
            row_h = max(x["h"] for x in current)
            if abs(it["y"] - row_y) > 0.65 * max(it["h"], row_h):
                rows.append(current)
                current = []
        current.append(it)
    if current:
        rows.append(current)
    return [it for row in rows for it in sorted(row, key=lambda it: it["x"])]


def is_marker(text, letter, strict):
    m = MARKER.match(text)
    if not m or m.group(1) != letter:
        return None
    paren, rest = m.group(2), m.group(3)
    if paren:
        return rest
    if strict:
        return None
    # Without a parenthesis, accept "C" alone, "E16"-style values, or single-letter values
    if rest == "" or re.match(r"^[-+−\d]", rest) or len(rest.strip()) == 1:
        return rest
    return None


def clean_value(val):
    val = val.lstrip(")）] ").strip()
    val = val.replace("−", "-")
    # Strip outer parentheses around single circled letters, e.g. "(A)" -> "A"
    m = re.match(r"^[\(\[]([A-Za-z0-9])[\)\]]$", val)
    if m:
        val = m.group(1)
    return val.strip()


def parse_options(boxes):
    """Return {"A": "12", ..., "E": "24"}, or {} when the options are not text."""
    items = reading_order(boxes)
    starts = [i for i, it in enumerate(items) if is_marker(it["text"], "A", strict=True) is not None]
    for start in reversed(starts):  # the options are the last A..E sequence
        options, current, i = {}, None, start
        expected = iter(LETTERS)
        letter = next(expected)
        while i < len(items):
            text = items[i]["text"]
            rest = is_marker(text, letter, strict=(letter == "A")) if letter else None
            if rest is not None:
                current = letter
                options[current] = rest
                letter = next(expected, None)
            elif current:
                options[current] = (options[current] + " " + text).strip()
            i += 1
        if len(options) == 5:
            cleaned = {k: clean_value(v) for k, v in options.items()}
            if all(cleaned.values()):
                return cleaned
            # If all 5 markers A)..E) were detected on the same horizontal row,
            # empty values are usually circled letters whose circle wasn't OCR'd.
            marker_ys = [items[j]["y"] for j in range(start, len(items))
                         if MARKER.match(items[j]["text"])]
            if len(marker_ys) >= 5 and (max(marker_ys) - min(marker_ys)) < 25:
                filled = {k: (v if v else k) for k, v in cleaned.items()}
                if all(filled.values()):
                    return filled
            return {}  # five markers, but the options are pictures
    return {}


def extract_options(ocr, image_bytes):
    """Run OCR on the bottom region first (fast & clean), falling back to the full image."""
    bottom_img = preprocess_image(image_bytes, crop_bottom=True)
    boxes, _ = ocr(bottom_img)
    options = parse_options(boxes or [])
    if options:
        return options

    full_img = preprocess_image(image_bytes, crop_bottom=False)
    boxes, _ = ocr(full_img)
    return parse_options(boxes or [])


def parse_options_2d(boxes):
    """Fallback parser for vertical lists (A..E stacked) and 2-row fraction layouts.
    Finds strict A)..E) markers and assigns nearby boxes by 2D proximity."""
    if not boxes:
        return {}
    items = []
    for box, text, _score in boxes:
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        items.append({
            "x": min(xs),
            "x_max": max(xs),
            "cx": (min(xs) + max(xs)) / 2,
            "y": (min(ys) + max(ys)) / 2,
            "h": max(ys) - min(ys),
            "text": text.strip(),
        })

    # Find strict markers A), B), C), D), E)
    markers = {}
    for idx, it in enumerate(items):
        m = MARKER.match(it["text"])
        if m and m.group(2):  # requires explicit delimiter like ) or ] or .
            letter = m.group(1)
            # Keep the lowest/leftmost occurrence of each letter
            if letter not in markers or it["y"] > markers[letter]["y"]:
                markers[letter] = {**it, "rest": m.group(3).strip(), "idx": idx}

    if len(markers) != 5 or set(markers.keys()) != set(LETTERS):
        return {}

    # Sanity check: markers must appear in valid reading order (top-to-bottom or left-to-right)
    pts = [markers[c] for c in LETTERS]
    for i in range(4):
        dy = pts[i + 1]["y"] - pts[i]["y"]
        dx = pts[i + 1]["x"] - pts[i]["x"]
        if dy < -15 and dx < 20:
            return {}

    used_indices = {markers[c]["idx"] for c in LETTERS}
    min_marker_y = min(p["y"] for p in pts) - 2.5 * max(p["h"] for p in pts)
    min_marker_x = min(p["x"] for p in pts) - 10

    buckets = {c: ([markers[c]["rest"]] if markers[c]["rest"] else []) for c in LETTERS}
    bucket_items = {c: [] for c in LETTERS}

    for idx, it in enumerate(items):
        if idx in used_indices:
            continue
        # Ignore question text above the option block or far to the left
        if it["y"] < min_marker_y or it["cx"] < min_marker_x:
            continue
        # Find nearest marker that is to the left of (or aligned with) this box
        best_c, best_dist = None, float("inf")
        for c in LETTERS:
            m = markers[c]
            if it["cx"] < m["x"] - 5:
                continue
            dx = max(0.0, it["x"] - m["x_max"])
            dy = abs(it["y"] - m["y"])
            # Weight vertical distance higher so rows don't bleed into each other
            dist = dx + 3.0 * dy
            if dist < best_dist:
                best_dist = dist
                best_c = c
        if best_c is not None:
            bucket_items[best_c].append(it)

    for c in LETTERS:
        # Sort fragments inside each option top-to-bottom (for fractions), then left-to-right
        frags = sorted(bucket_items[c], key=lambda it: (round(it["y"] / 12), it["x"]))
        extra = " ".join(f["text"] for f in frags)
        full_val = " ".join(buckets[c] + ([extra] if extra else [])).strip()
        buckets[c] = clean_value(full_val)

    if all(buckets.values()):
        return buckets
    return {}


def extract_options_fallback(ocr, image_bytes):
    """Only runs when extract_options() returns {}. Crops out right-side diagrams
    and uses 2D spatial matching for vertical option lists and stacked fractions."""
    from PIL import Image, ImageOps
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    # Crop left 68% of image to strip right-side geometry diagrams, then upscale 2x
    left_img = img.crop((0, 0, int(w * 0.68), h))
    lw, lh = left_img.size
    left_img = left_img.resize((lw * 2, lh * 2), Image.Resampling.BICUBIC)
    left_img = ImageOps.expand(left_img, border=32, fill="white")
    buf = io.BytesIO()
    left_img.save(buf, format="PNG")

    boxes, _ = ocr(buf.getvalue())
    opts = parse_options_2d(boxes or [])
    if opts:
        return opts

    # Second try on full upscaled image (for wide horizontal options like Image 1)
    full_img = preprocess_image(image_bytes, crop_bottom=False)
    boxes_full, _ = ocr(full_img)
    return parse_options_2d(boxes_full or [])


def main():
    args = parse_args()
    import pyarrow.parquet as pq

    languages = set(args.languages.split(","))
    val_years = set(y for y in args.val_years.split(",") if y)
    os.makedirs(args.out, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    ocr = None
    if not args.no_ocr:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()

    cache_path = os.path.join(args.out, ".ocr_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                row = json.loads(line)
                cache[row["key"]] = row["options"]
    cache_file = open(cache_path, "a")

    rows = []
    with_options_count = 0
    t0 = time.time()
    for path in shard_paths(args):
        table = pq.read_table(path, columns=["language"])
        wanted = [i for i, lang in enumerate(table.column(0).to_pylist()) if lang in languages]
        if not wanted:
            continue
        full = pq.read_table(path).take(wanted)
        for i in range(full.num_rows):
            if args.limit and len(rows) >= args.limit:
                break
            r = {c: full.column(c)[i].as_py() for c in ("id", "language", "level", "text_only", "label", "text")}
            key = f"{r['id']}|{r['language']}"
            image = full.column("image")[i].as_py()["bytes"]
            if args.save_images:
                with open(os.path.join(args.out, "images", key.replace("|", "_") + ".png"), "wb") as f:
                    f.write(image)
            if key in cache:
                options = cache[key]
            elif ocr:
                options = extract_options(ocr, image)
                if not options:
                    options = extract_options_fallback(ocr, image)
                cache[key] = options
                cache_file.write(json.dumps({"key": key, "options": options}, ensure_ascii=False) + "\n")
            else:
                options = {}
            if options:
                with_options_count += 1
            year = r["id"].split("_")[1]
            if val_years:
                split = "val" if year in val_years else "train"
            else:
                bucket = int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 1000
                split = "val" if bucket < args.val_fraction * 1000 else "train"
            rows.append({
                "id": r["id"], "language": r["language"], "level": r["level"], "year": int(year),
                "text_only": r["text_only"], "text": r["text"].strip(), "options": options,
                "label": r["label"], "split": split,
            })
            if len(rows) % 100 == 0:
                rate = len(rows) / (time.time() - t0)
                pct = 100.0 * with_options_count / len(rows)
                print(
                    f"  {len(rows)} rows, {rate:.1f} rows/s | "
                    f"OCR success: {with_options_count}/{len(rows)} ({pct:.1f}%)",
                    flush=True,
                )
    cache_file.close()

    for split in ("train", "val"):
        out = os.path.join(args.out, f"{split}.jsonl")
        with open(out, "w") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "split"}, ensure_ascii=False) + "\n")

    report(rows, args.out)


def report(rows, out):
    if not rows:
        sys.exit("No rows matched. Check --languages and --shards.")
    by = Counter((r["split"], r["language"]) for r in rows)
    with_options = sum(1 for r in rows if r["options"])
    text_only = [r for r in rows if r["text_only"]]
    labels = Counter(r["label"] for r in rows if r["split"] == "train")
    top, count = labels.most_common(1)[0]
    print()
    print(f"Wrote {out}/train.jsonl and {out}/val.jsonl")
    for (split, lang), n in sorted(by.items()):
        print(f"  {split:5} {lang}: {n}")
    print(f"Options read from the image: {with_options}/{len(rows)} ({100 * with_options / len(rows):.0f}%)")
    if text_only:
        ok = sum(1 for r in text_only if r["options"])
        print(f"  of which text-only problems: {ok}/{len(text_only)} ({100 * ok / len(text_only):.0f}%)")
    print(f"Most common answer in train: {top} ({100 * count / sum(labels.values()):.1f}%). "
          f"Always answering {top} is the floor to beat.")


if __name__ == "__main__":
    main()
