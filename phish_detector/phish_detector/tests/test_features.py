from phish_detector.features import get_feature_pipeline

def test_pipeline_runs():
    pipeline = get_feature_pipeline()
    sample = ["http://example.com", "http://phishy.com/login"]
    X = pipeline.fit_transform(sample)
    assert X.shape[0] == 2

