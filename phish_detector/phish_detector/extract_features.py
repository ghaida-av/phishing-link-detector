import pandas as pd
from phish_detector.features import get_feature_pipeline

if __name__ == "__main__":
    # Example dataset
    data = {
        "url": ["http://example.com", "http://phishy.com/login"],
        "label": [0, 1]
    }
    df = pd.DataFrame(data)

    pipeline = get_feature_pipeline()
    X = pipeline.fit_transform(df["url"])

    print("Feature matrix shape:", X.shape)

