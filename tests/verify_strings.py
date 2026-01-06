
import json
import os

def check_strings_json():
    path = "/root/ha_washdata/custom_components/ha_washdata/strings.json"
    with open(path, "r") as f:
        data = json.load(f)

    errors = []

    def check_step(step_name, step_content):
        if "data_description" in step_content:
            data_desc = step_content["data_description"]
            data_keys = step_content.get("data", {}).keys()
            for key in data_desc:
                if key not in data_keys:
                    errors.append(f"Step '{step_name}': Key '{key}' in data_description but not in data.")

    # Check config steps
    config = data.get("config", {})
    for step_name, step_content in config.get("step", {}).items():
        check_step(f"config.step.{step_name}", step_content)

    # Check options steps
    options = data.get("options", {})
    for step_name, step_content in options.get("step", {}).items():
        check_step(f"options.step.{step_name}", step_content)

    if errors:
        print("Validation FAILED:")
        for e in errors:
            print(f" - {e}")
        exit(1)
    else:
        print("Validation PASSED: All data_description keys match data keys.")

if __name__ == "__main__":
    check_strings_json()
