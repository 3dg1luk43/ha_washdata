import argparse
import json
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
    Returns a set of keys that have changed in the given file compared to HEAD~1.
    This includes added and modified keys.
    """
    try:
        # Get the previous version of the file content
        cmd = ["git", "show", f"HEAD~1:{file_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        prev_content = json.loads(result.stdout)

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
        "--card-js",
        dest="card_js",
        help="Path to the frontend card JS file to update translations in.",
    )

    args = argparser.parse_args()

    translations_dir = Path(args.translations_dir)
    en_file = translations_dir / "en.json"
    strings_file = translations_dir.parent / "strings.json"

    # Sync en.json with strings.json if it exists (canonical source)
    if strings_file.exists():
        print(f"Syncing {en_file} with {strings_file}...")
        strings_data = load_json(strings_file)
        save_json(en_file, strings_data)

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

    if not languages and not args.card_js:
        print("No languages to process and no card update requested.")
        return 2

    print(f"Processing translations for {', '.join(languages)} in {translations_dir}")

    retranslate = args.retranslate

    # 1. Identify changed keys in en.json
    # We always load en_flat as the source of truth
    en_content = load_json(en_file)
    en_flat = flatten_dict(en_content)
    changed_keys_in_en, _ = get_git_diff_keys(str(en_file))
    print(f"Found {len(changed_keys_in_en)} changed/added keys in en.json via git diff")

    # 2. Iterate over all other json files
    for lang_code in languages:
        file = translations_dir / f"{lang_code}.json"
        target_lang_api = LANG_API_MAP.get(lang_code, lang_code)

        if not target_lang_api:
            print(f"{lang_code} not supported")
            continue

        print(f"Processing {lang_code} (API: {target_lang_api})...")

        new_file = retranslate
        target_flat = {}
        if not retranslate and file.exists():
            try:
                target_data = load_json(file)
                target_flat = flatten_dict(target_data)
            except Exception:
                print(f"Error decoding {file}, starting fresh.")
                new_file = True
        else:
            new_file = True

        keys_to_translate = []
        original_texts_to_translate = []

        for k, en_text in en_flat.items():
            # Translate if key is missing OR if it was changed in en.json OR if we are retranslating
            if k not in target_flat or k in changed_keys_in_en or retranslate:
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
            batch_size = 20
            for i in range(0, len(keys_to_translate), batch_size):
                batch_keys = keys_to_translate[i : i + batch_size]
                batch_texts = original_texts_to_translate[i : i + batch_size]

                print(f"    Batch {i//batch_size + 1}: Translating {len(batch_keys)} items...")
                translated_texts = translate_batch(batch_texts, target_lang_api)

                if translated_texts is None:
                    print(f"    Translation failed for batch. Skipping update for {lang_code}.")
                    break

                for k, b_text, t_text in zip(batch_keys, batch_texts, translated_texts):
                    # Restore arguments {device} etc
                    b_matches = ARGUMENTS_REGEX.findall(b_text)
                    t_matches = ARGUMENTS_REGEX.findall(t_text)
                    for index, match in enumerate(t_matches):
                        if index < len(b_matches) and b_matches[index] != match:
                            t_text = t_text.replace(match, b_matches[index], 1)
                    target_flat[k] = t_text

                time.sleep(0.5)
            else:
                final_data = unflatten_dict(target_flat)
                save_json(file, final_data)
                print(f"  {'Created' if new_file else 'Updated'} {file}")
        else:
            final_data = unflatten_dict(target_flat)
            save_json(file, final_data)
            print(f"  Updated {file}")

    # 3. Update frontend translations if requested
    if args.card_js:
        card_js_path = Path(args.card_js)
        if card_js_path.exists():
            print(f"Updating frontend translations in {card_js_path}...")
            en_card = en_content.get("card", {})
            frontend_translations = {"en": en_card}
            
            # Use all languages that have a json file, or just all if --all was used
            possible_languages = set(languages) | {"en"}
            for f in translations_dir.glob("*.json"):
                possible_languages.add(f.stem)

            for lang_code in sorted(possible_languages):
                if lang_code == "en": continue
                lang_file = translations_dir / f"{lang_code}.json"
                
                # Default to English card keys if translation is missing
                lang_card = en_card.copy()
                
                if lang_file.exists():
                    try:
                        data = load_json(lang_file)
                        if "card" in data:
                            # Overlay translated keys onto English defaults
                            for k, v in data["card"].items():
                                lang_card[k] = v
                    except Exception as e:
                        print(f"Warning: Could not read {lang_file}: {e}")
                
                frontend_translations[lang_code] = lang_card
            
            if frontend_translations:
                with open(card_js_path, "r", encoding="utf-8") as f:
                    js_content = f.read()
                
                js_translations = json.dumps(frontend_translations, indent=2, ensure_ascii=False)
                pattern = re.compile(r"const TRANSLATIONS = \{.*?\};", re.DOTALL)
                
                if pattern.search(js_content):
                    # Use a lambda for replacement to avoid backslash escaping issues in re.sub
                    new_js_content = pattern.sub(lambda _: f"const TRANSLATIONS = {js_translations};", js_content)
                    with open(card_js_path, "w", encoding="utf-8") as f:
                        f.write(new_js_content)
                    print(f"Successfully updated frontend translations in {card_js_path}")
                else:
                    print(f"Warning: Could not find 'const TRANSLATIONS = {{...}};' in {card_js_path}")
        else:
            print(f"Warning: Card JS file {card_js_path} not found.")

    return 0


if __name__ == "__main__":
    sys.exit(main())