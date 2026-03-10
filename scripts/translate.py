import argparse
import json
import os
import re
import requests
import subprocess
import sys
import time
from pathlib import Path

from deep_translator import GoogleTranslator

# No API KEY needed for deep-translator (Google v2 wrapper)

# List of supported languages by Home Assistant
LANGUAGES_URL = "https://raw.githubusercontent.com/home-assistant/frontend/dev/src/translations/translationMetadata.json"

LANG_API_MAP = {
    "en-GB": None,  # British English
    "es-419": "es",  # Latin American Spanish
    "gsw": "de",  # Swiss German
    "he": "iw",  # Hebrew
    # "kw": None,  # Cornish, not supported by Google Translate
    "nb": "no",  # Norwegian Bokmål
    "nn": None,  # Norwegian Nynorsk
    "pt-BR": "pt",  # Brazilian Portuguese
    "sr": None,  # Serbian in Cyrillic, not supported by Google Translate
    "sr-Latn": "sr",  # Serbian in Latin
    # "zh-HK": "zh-CN",  # Hong Kong Chinese
    "zh-Hans": "zh-CN",  # Simplified Chinese
    "zh-Hant": "zh-TW",  # Traditional Chinese 
}

ARGUMENTS_REGEX = re.compile("{.*?}")
TRANSLATIONS_MARKER = "const TRANSLATIONS ="


def is_ci_environment():
    return os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n') # Add trailing newline


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='.'):
    result = {}
    for k, v in d.items():
        parts = k.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = v
    return result


def get_git_diff_keys(file_path):
    """
    Returns a set of keys that have changed in the given file compared to HEAD.
    This includes added and modified keys.
    """
    try:
        # Get the previous version of the file content
        git_file_path = get_git_relative_path(file_path)
        prev_content_text = get_git_baseline_content(git_file_path)
        if prev_content_text is None:
            print("Warning: No git baseline found. Treating all en.json keys as changed.")
            curr_flat = flatten_dict(load_json(file_path))
            return set(curr_flat.keys()), curr_flat

        prev_content = json.loads(prev_content_text)

        # Get current version
        curr_content = load_json(file_path)

        prev_flat = flatten_dict(prev_content)
        curr_flat = flatten_dict(curr_content)

        changed_keys = set()

        # Check for modified or added values in current version
        for k, v in curr_flat.items():
            if k not in prev_flat:
                # New key
                changed_keys.add(k)
            elif prev_flat[k] != v:
                # Modified value
                changed_keys.add(k)

        return changed_keys, curr_flat
    except subprocess.CalledProcessError:
        print("Warning: Could not get git diff. Assuming all keys might need check if missing.")
        return set(), flatten_dict(load_json(file_path))
    except Exception as e:
        print(f"Error checking git diff: {e}")
        return set(), flatten_dict(load_json(file_path))


def get_git_relative_path(file_path):
    """Return file_path relative to git root when possible."""
    try:
        root_cmd = ["git", "rev-parse", "--show-toplevel"]
        root_result = subprocess.run(root_cmd, capture_output=True, text=True, check=True)
        git_root = Path(root_result.stdout.strip())
        return str(Path(file_path).resolve().relative_to(git_root))
    except Exception:
        return file_path


def try_git_show(refspec):
    try:
        cmd = ["git", "show", refspec]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def get_git_baseline_content(git_file_path):
    """
    Resolve baseline file content for diffing.
    CI: strictly compare against HEAD.
    Local: prefer HEAD, then index snapshot; fallback to None.
    """
    if is_ci_environment():
        return try_git_show(f"HEAD:{git_file_path}")

    head_content = try_git_show(f"HEAD:{git_file_path}")
    if head_content is not None:
        return head_content

    # Local fallback: when file is staged but not available in HEAD.
    return try_git_show(f":{git_file_path}")


def translate_batch(texts, target_lang):
    """
    Translates a list of texts to the target language using deep-translator.
    """
    try:
        # deep-translator's GoogleTranslator supports batching (list of strings)
        translator = GoogleTranslator(source='en', target=target_lang)

        # It handles batching internally or via the library, but let's be safe
        # The library documentation says .translate_batch(batch)
        translations = translator.translate_batch(texts)

        if len(translations) != len(texts):
            print(f"Warning: Mismatch in translation count for {target_lang}")
            return None

        return translations

    except Exception as e:
        print(f"Error translating to {target_lang}: {e}")
        return None


def translate_text(text, target_lang):
    """Translate a single text item."""
    try:
        translator = GoogleTranslator(source='en', target=target_lang)
        translated = translator.translate(text)
        if translated is None:
            return text
        return translated
    except Exception as e:
        print(f"      Item translation failed for {target_lang}: {e}")
        return text


def translate_batch_resilient(texts, target_lang):
    """Translate in batch first, then fall back to per-item translation on failure."""
    translations = translate_batch(texts, target_lang)
    if translations is not None:
        return translations

    print(f"    Falling back to per-item translation for {target_lang}...")
    return [translate_text(text, target_lang) for text in texts]


def extract_js_object(content, marker):
    """Extract an object literal that starts after marker and return (start, end, text)."""
    marker_pos = content.find(marker)
    if marker_pos == -1:
        raise ValueError(f"Marker not found: {marker}")

    start = content.find("{", marker_pos)
    if start == -1:
        raise ValueError(f"Object start not found after marker: {marker}")

    depth = 0
    in_string = False
    string_char = ""
    escaped = False

    for idx in range(start, len(content)):
        char = content[idx]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_char:
                in_string = False
            continue

        if char in ('"', "'"):
            in_string = True
            string_char = char
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return start, idx, content[start: idx + 1]

    raise ValueError("Could not find end of JS object literal")


def parse_card_translations(file_path):
    content = Path(file_path).read_text(encoding="utf-8")
    start, end, object_text = extract_js_object(content, TRANSLATIONS_MARKER)

    # The card object is JSON-like. Remove trailing commas for JSON parsing.
    json_like_text = re.sub(r",(\s*[}\]])", r"\1", object_text)
    translations = json.loads(json_like_text)
    return content, start, end, translations


def get_card_en_diff_keys(file_path):
    """Return changed keys in TRANSLATIONS.en vs HEAD and current parsed card translations."""
    _, _, _, curr_translations = parse_card_translations(file_path)
    curr_en = curr_translations.get("en", {})

    try:
        git_file_path = get_git_relative_path(file_path)
        prev_content_text = get_git_baseline_content(git_file_path)
        if prev_content_text is None:
            print("Warning: No git baseline found for card. Treating all card en keys as changed.")
            return set(curr_en.keys()), curr_translations

        _, _, _, prev_translations = parse_card_translations_from_content(prev_content_text)
        prev_en = prev_translations.get("en", {})

        changed_keys = set()
        for key, value in curr_en.items():
            if key not in prev_en or prev_en[key] != value:
                changed_keys.add(key)

        return changed_keys, curr_translations
    except subprocess.CalledProcessError:
        print("Warning: Could not get git diff for card translations.")
        return set(), curr_translations
    except Exception as exc:
        print(f"Error checking card translation git diff: {exc}")
        return set(), curr_translations


def parse_card_translations_from_content(content):
    start, end, object_text = extract_js_object(content, TRANSLATIONS_MARKER)
    json_like_text = re.sub(r",(\s*[}\]])", r"\1", object_text)
    translations = json.loads(json_like_text)
    return content, start, end, translations


def apply_placeholder_fixes(source_text, translated_text):
    source_matches = ARGUMENTS_REGEX.findall(source_text)
    translated_matches = ARGUMENTS_REGEX.findall(translated_text)
    for idx, translated_match in enumerate(translated_matches):
        if idx < len(source_matches) and source_matches[idx] != translated_match:
            translated_text = translated_text.replace(translated_match, source_matches[idx])
    return translated_text


def update_language_map(en_map, target_map, changed_keys, retranslate, lang_code, target_lang_api):
    """Update one language map based on en_map and return (updated_map, did_change, success)."""
    updated_map = dict(target_map)
    did_change = False

    keys_to_remove = [k for k in updated_map if k not in en_map]
    if keys_to_remove:
        print(f"  Removing {len(keys_to_remove)} deleted keys.")
        for key in keys_to_remove:
            del updated_map[key]
        did_change = True

    keys_to_translate = []
    texts_to_translate = []
    for key, en_text in en_map.items():
        if not isinstance(en_text, str):
            continue
        if retranslate or key not in updated_map or key in changed_keys:
            keys_to_translate.append(key)
            texts_to_translate.append(en_text)

    if not keys_to_translate:
        return updated_map, did_change, True

    print(f"  Translating {len(keys_to_translate)} keys to {lang_code}...")

    batch_size = 20
    for i in range(0, len(keys_to_translate), batch_size):
        batch_keys = keys_to_translate[i: i + batch_size]
        batch_texts = texts_to_translate[i: i + batch_size]
        print(f"    Batch {i // batch_size + 1}: Translating {len(batch_keys)} items...")

        translated_texts = translate_batch_resilient(batch_texts, target_lang_api)

        for key, source_text, translated_text in zip(batch_keys, batch_texts, translated_texts):
            updated_map[key] = apply_placeholder_fixes(source_text, translated_text)
            did_change = True

        time.sleep(1.0)

    return updated_map, did_change, True


def process_card_translations(card_file_path, languages, retranslate):
    if not card_file_path.exists():
        print(f"Card file not found, skipping card translations: {card_file_path}")
        return

    print(f"Processing card translations in {card_file_path}")
    changed_keys_in_en, card_translations = get_card_en_diff_keys(str(card_file_path))
    en_map = card_translations.get("en", {})

    if not isinstance(en_map, dict) or not en_map:
        print("Card translations missing canonical 'en' map, skipping.")
        return

    print(f"Found {len(changed_keys_in_en)} changed/added keys in card en translations")

    updated_any = False
    for lang_code in languages:
        target_lang_api = LANG_API_MAP.get(lang_code, lang_code)
        if not target_lang_api:
            print(f"{lang_code} not supported")
            continue

        print(f"Processing card {lang_code} (API: {target_lang_api})...")
        current_target_map = card_translations.get(lang_code, {})
        if not isinstance(current_target_map, dict):
            current_target_map = {}

        updated_map, did_change, success = update_language_map(
            en_map=en_map,
            target_map=current_target_map,
            changed_keys=changed_keys_in_en,
            retranslate=retranslate,
            lang_code=lang_code,
            target_lang_api=target_lang_api,
        )

        if not success:
            continue

        if did_change or lang_code not in card_translations:
            card_translations[lang_code] = updated_map
            updated_any = True
            print(f"  Updated card language: {lang_code}")
        else:
            print(f"  No changes needed for card {lang_code}.")

    if not updated_any:
        print("No card translation changes needed.")
        return

    content, start, end, _ = parse_card_translations(str(card_file_path))
    new_object_text = json.dumps(card_translations, indent=2, ensure_ascii=False)
    new_content = content[:start] + new_object_text + content[end + 1:]
    Path(card_file_path).write_text(new_content, encoding="utf-8")
    print(f"Updated card translations in {card_file_path}")


def remove_unsupported_translation_files(translations_dir, supported_languages):
    """Delete language json files that are not in the supported language set."""
    removed_count = 0
    for file in translations_dir.glob("*.json"):
        lang_code = file.stem
        if lang_code not in supported_languages:
            file.unlink()
            removed_count += 1
            print(f"Removed unsupported language file: {file}")
    if removed_count == 0:
        print("No unsupported translation files to remove.")


def prune_unsupported_card_languages(card_file_path, supported_languages):
    """Remove unsupported language entries from card TRANSLATIONS object."""
    if not card_file_path.exists():
        return

    try:
        content, start, end, translations = parse_card_translations(str(card_file_path))
    except Exception as exc:
        print(f"Warning: Could not parse card file for pruning: {exc}")
        return

    original_keys = set(translations.keys())
    allowed = set(supported_languages)
    allowed.add("en")

    for key in list(translations.keys()):
        if key not in allowed:
            del translations[key]

    if set(translations.keys()) == original_keys:
        print("No unsupported card languages to remove.")
        return

    new_object_text = json.dumps(translations, indent=2, ensure_ascii=False)
    new_content = content[:start] + new_object_text + content[end + 1:]
    Path(card_file_path).write_text(new_content, encoding="utf-8")
    print(f"Removed unsupported card languages in {card_file_path}")


def main():
    # Read command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "translations_dir", help="The directory containing the translation files."
    )
    argparser.add_argument(
        "languages",
        nargs="*",
        help="Space-separated list of languages to translate to.",
    )
    argparser.add_argument(
        "--all",
        dest="all_languages",
        action="store_true",
        help="Generate translations for all by Home Assistant supported languages.",
    )
    argparser.add_argument(
        "--retranslate",
        dest="retranslate",
        action="store_true",
        help="Retranslate already translated keys.",
    )
    argparser.add_argument(
        "--card-file",
        dest="card_file",
        help=(
            "Optional path to the card JS file containing const TRANSLATIONS; "
            "defaults to <translations_dir>/../www/ha-washdata-card.js"
        ),
    )

    args = argparser.parse_args()

    translations_dir = Path(args.translations_dir)
    en_file = translations_dir / "en.json"

    if not en_file.exists():
        print(f"Error: {en_file} not found.")
        return 1

    response = requests.get(LANGUAGES_URL)
    ha_languages = json.loads(response.content)
    ha_languages = set(ha_languages.keys())

    if args.all_languages:
        # Generate translations for all by Home Assistant supported languages.
        languages = ha_languages
    elif len(args.languages) > 0:
        # Generate translations for the languages given as arguments and supported by Home Assistant.
        # Compare the languages case insensitive to the list of languages supported by Home Assistant and use the propperly cased language from this list.
        lc_languages = {l.lower() for l in args.languages}
        languages = [
            language
            for language in ha_languages
            if language.lower() in lc_languages
        ]
    else:
        # Take the languages from the already generated language files.
        languages = [file.stem for file in translations_dir.glob("*.json")]

    # Remove duplicate languages, if any, and sort alphabetically.
    languages = sorted(list(set(languages)))

    # Remove the English language as this is the source language.
    if "en" in languages:
        languages.remove("en")

    if len(languages) == 0:
        return 2

    print(f"Processing translations for {', '.join(languages)} in {translations_dir}")

    retranslate = args.retranslate

    # 1. Identify changed keys in en.json
    changed_keys_in_en, en_flat = get_git_diff_keys(str(en_file))
    print(f"Found {len(changed_keys_in_en)} changed/added keys in en.json")

    # 2. Iterate over all other json files
    for lang_code in languages:
        file = translations_dir / f"{lang_code}.json"

        # Map HA language codes to Google Translate codes if necessary
        # deep-translator uses ISO 639-1 mostly.
        # Common mappings:
        target_lang_api = LANG_API_MAP.get(lang_code, lang_code)

        if not target_lang_api:
            print(f"{lang_code} not supported")
            continue

        print(f"Processing {lang_code} (API: {target_lang_api})...")

        new_file = retranslate
        target_data = {}
        target_flat = {}
        if not retranslate:
            try:
                target_data = load_json(file)
                target_flat = flatten_dict(target_data)
            except FileNotFoundError:
                # Language file does not exist yet, nothing to worry about.
                new_file = True
            except Exception:
                print(f"Error decoding {file}, starting fresh.")

        keys_to_translate = []
        original_texts_to_translate = []

        for k, en_text in en_flat.items():
            if k not in target_flat:
                keys_to_translate.append(k)
                original_texts_to_translate.append(en_text)
            elif k in changed_keys_in_en:
                keys_to_translate.append(k)
                original_texts_to_translate.append(en_text)

        keys_to_remove = [k for k in target_flat if k not in en_flat]
        if keys_to_remove:
            print(f"  Removing {len(keys_to_remove)} deleted keys.")
            for k in keys_to_remove:
                del target_flat[k]

        if not keys_to_translate and not keys_to_remove:
            print(f"  No changes needed for {lang_code}.")
            continue

        if keys_to_translate:
            print(f"  Translating {len(keys_to_translate)} keys to {lang_code}...")

            # Batch translate
            # deep-translator might have smaller limits per request or rate limits
            # batch_size=20 is safe
            batch_size = 20
            for i in range(0, len(keys_to_translate), batch_size):
                batch_keys = keys_to_translate[i : i + batch_size]
                batch_texts = original_texts_to_translate[i : i + batch_size]

                print(f"    Batch {i//batch_size + 1}: Translating {len(batch_keys)} items...")
                translated_texts = translate_batch_resilient(batch_texts, target_lang_api)

                for k, b_text, t_text in zip(batch_keys, batch_texts, translated_texts):
                    target_flat[k] = apply_placeholder_fixes(b_text, t_text)

                # Rate limiting logic for free API
                time.sleep(1.0)

            final_data = unflatten_dict(target_flat)
            save_json(file, final_data)
            if new_file:
                print(f"  Created {file}")
            else:
                print(f"  Updated {file}")
        else:
            # Only removals happened, save it
            final_data = unflatten_dict(target_flat)
            save_json(file, final_data)
            print(f"  Updated {file}")

    card_file = (
        Path(args.card_file)
        if args.card_file
        else translations_dir.parent / "www" / "ha-washdata-card.js"
    )

    # In --all mode, enforce HA language list as source of truth and remove extra locales.
    if args.all_languages:
        supported_with_en = set(ha_languages)
        supported_with_en.add("en")
        remove_unsupported_translation_files(translations_dir, supported_with_en)
        prune_unsupported_card_languages(card_file, supported_with_en)

    process_card_translations(card_file, languages, retranslate)

    return 0


if __name__ == "__main__":
    sys.exit(main())
