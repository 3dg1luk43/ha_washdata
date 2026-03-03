import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from deep_translator import GoogleTranslator

# No API KEY needed for deep-translator (Google v2 wrapper)

# List of supported languages by Home Assistant
# Taken from https://developers.home-assistant.io/docs/voice/intent-recognition/supported-languages/
HA_LANGUAGES = [
    "af",
    "ar",
    "bg",
    "bn",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "de-CH",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "ga",
    "gl",
    "gu",
    "he",
    "hi",
    "hr",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "ka",
    "kn",
    "ko",
    "kw",
    "lb",
    "lt",
    "lv",
    "ml",
    "mn",
    "mr",
    "ms",
    "nb",
    "ne",
    "nl",
    "pa",
    "pl",
    "pt",
    "pt-BR",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sr-Latn",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh-CN",
    "zh-HK",
    "zh-TW",
]

LANG_API_MAP = {
    "de-CH": "de",  # Swiss German
    "he": "iw",  # Hebrew
    "kw": None,  # Cornish, not supported by Google Translate
    "nb": "no",  # Norwegian Bokmål
    "pt-BR": "pt",  # Brazilian Portuguese
    "sr": None,  # Serbian in Cyrillic, not supported by Google Translate
    "sr-Latn": "sr",  # Serbian in Latin
    "zh-HK": "zh-CN",  # Hong Kong Chinese
}


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

    args = argparser.parse_args()

    translations_dir = Path(args.translations_dir)
    en_file = translations_dir / "en.json"

    if not en_file.exists():
        print(f"Error: {en_file} not found.")
        return 1

    if args.all_languages:
        # Generate translations for all by Home Assistant supported languages.
        languages = HA_LANGUAGES
    elif len(args.languages) > 0:
        # Generate translations for the languages given as arguments and supported by Home Assistant.
        # Compare the languages case insensitive to the list of languages supported by Home Assistant and use the propperly cased language from this list.
        languages = [
            language
            for language in HA_LANGUAGES
            if language.lower() in (l.lower() for l in args.languages)
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
        target_lang_api = lang_code
        if lang_code in LANG_API_MAP:
            target_lang_api = LANG_API_MAP[lang_code]

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
                translated_texts = translate_batch(batch_texts, target_lang_api)

                if translated_texts is None:
                    print(f"    Translation failed for batch. Skipping update for {lang_code}.")
                    break

                for k, t_text in zip(batch_keys, translated_texts):
                    target_flat[k] = t_text

                # Rate limiting logic for free API
                time.sleep(1.0)
            else:
                # Only save if loop finished normally
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

    return 0

if __name__ == "__main__":
    sys.exit(main())
