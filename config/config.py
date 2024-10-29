import os
import sys
import yaml

file_path = "configurations.yaml"


def load_config():
    if getattr(sys, 'frozen', False):
        # Running as a bundled executable
        script_dir = sys._MEIPASS
    else:
        # Running as a script
        script_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(script_dir, file_path)

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    return config


if __name__ == "__main__":
    config = load_config()
    print(config)
