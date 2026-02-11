import json
import subprocess
import time
from pathlib import Path
from deep_translator import GoogleTranslator

# No API KEY needed for deep-translator (Google v2 wrapper)

TRANSLATIONS_DIR = Path("custom_components/ha_washdata/translations")
EN_FILE = TRANSLATIONS_DIR / "en.json"

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
    if not EN_FILE.exists():
        print(f"Error: {EN_FILE} not found.")
        return

    print(f"Processing translations in {TRANSLATIONS_DIR}")
    
    # 1. Identify changed keys in en.json
    changed_keys_in_en, en_flat = get_git_diff_keys(str(EN_FILE))
    print(f"Found {len(changed_keys_in_en)} changed/added keys in en.json")
    
    # 2. Iterate over all other json files
    for file in TRANSLATIONS_DIR.glob("*.json"):
        if file.name == "en.json":
            continue
            
        lang_code = file.stem
        # Map HA language codes to Google Translate codes if necessary
        # deep-translator uses ISO 639-1 mostly.
        # Common mappings:
        target_lang_api = lang_code
        if lang_code == "zh-Hans":
            target_lang_api = "zh-CN"
        elif lang_code == "zh-Hant":
            target_lang_api = "zh-TW"
        elif lang_code == "nb": # Norwegian Bokmål
             target_lang_api = "no"
        elif lang_code == "nl-BE":
             target_lang_api = "nl"
        
        print(f"Processing {lang_code} (API: {target_lang_api})...")
        
        try:
            target_data = load_json(file)
        except Exception:
            print(f"Error decoding {file}, starting fresh.")
            target_data = {}
            
        target_flat = flatten_dict(target_data)
        
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
                batch_keys = keys_to_translate[i:i+batch_size]
                batch_texts = original_texts_to_translate[i:i+batch_size]
                
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
                print(f"  Updated {file}")
        else:
             # Only removals happened, save it
             final_data = unflatten_dict(target_flat)
             save_json(file, final_data)
             print(f"  Updated {file}")

if __name__ == "__main__":
    main()
