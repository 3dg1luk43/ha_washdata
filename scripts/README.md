# Home Assistant Integration Translation Tool

A Python script to automatically translate Home Assistant integration strings and frontend card translations using Google Translate (via `deep-translator`).

## Features

- **Automated Translation**: Uses `deep-translator` (GoogleTranslator) to translate `en.json` to any language supported by Home Assistant.
- **No API Key Required**: Works without a Google Cloud Translate API key.
- **Smart Sync**: Automatically syncs `en.json` with `strings.json` if it exists.
- **Git Integration**: Uses `git diff` to identify only new or modified keys in `en.json` to minimize unnecessary translations.
- **Placeholder Preservation**: Correctly handles and preserves placeholders like `{device}`, `{state}`, etc.
- **Frontend Support**: Can update translations directly in a frontend Lovelace card JavaScript file.
- **Batch Processing**: Handles translations in batches with rate-limiting to avoid IP blocks.

## Requirements

- Python 3.6+
- `requests`
- `deep-translator`

You can install the dependencies via pip:

```bash
pip install requests deep-translator
```

## Usage

Run the script from the command line:

```bash
python3 translate.py <translations_dir> [languages...] [options]
```

### Arguments

- `translations_dir`: The directory containing the `.json` translation files (e.g., `custom_components/my_integration/translations/`).
- `languages`: (Optional) Space-separated list of language codes to translate to (e.g., `de fr es`). If omitted, it will process all existing `.json` files in the directory.

### Options

- `--all`: Generate translations for *all* languages supported by Home Assistant.
- `--retranslate`: Force retranslation of all keys, even if they already exist in the target language file.
- `--card-js <path>`: Path to a frontend card JS file. The script will update the `const TRANSLATIONS = { ... };` block in that file with the latest translations.

## Example

Translate all missing keys to German and French:
```bash
python3 translate.py ./custom_components/ha_washdata/translations de fr
```

Update all existing translation files with new keys from `en.json`:
```bash
python3 translate.py ./custom_components/ha_washdata/translations
```

Update both backend translations and the frontend card:
```bash
python3 translate.py ./custom_components/ha_washdata/translations --card-js ./dist/ha-washdata-card.js
```

## How it Works

1.  **Sync**: If `strings.json` exists in the parent directory, it overwrites `en.json`.
2.  **Diff**: It runs `git diff HEAD~1` on `en.json` to see which keys have changed.
3.  **Translate**: For each target language:
    - It loads existing translations.
    - It identifies keys that are missing or have changed in English.
    - It sends these keys to Google Translate in batches of 20.
    - It fixes any mangled curly-brace placeholders.
    - It merges the new translations and saves the file.
4.  **Frontend**: If `--card-js` is provided, it extracts the `card` section from all translation files and injects them into the specified JS file.
