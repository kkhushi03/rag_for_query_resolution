import yaml

def load_config(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("RETRIEVAL_METHOD", "tfidf")  # Defaults to tfidf if not specified
    return config

CONFIG = load_config()
