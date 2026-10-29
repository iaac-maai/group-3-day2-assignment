"""Predict one monthly rent; load only trusted joblib artifacts."""
import argparse
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

EXPECTED_SHA256 = "b4193b09cc1316008936387fd28f65b07960447feab5d1eaa279ed8745791262"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Use a local trusted artifact instead of downloading")
    parser.add_argument("--revision", default="main", help="Reviewed commit SHA or refs/pr/<number>")
    args = parser.parse_args()
    path = args.model or Path(hf_hub_download(
        repo_id="iaac-maai/group3-day2-mumbairental",
        filename="model.joblib", revision=args.revision,
    ))
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Artifact checksum mismatch; refusing to deserialize")
    model = joblib.load(path)
    example = pd.DataFrame([{
        "Area_num": 330.0, "BHK_num": 1.0, "Bathrooms_num": 1.0,
        "Balconies_num": np.nan, "Furnished_clean": "Furnished", "Facing": "East",
        "Location": "Mumbai", "Property Type": "Apartment",
    }])
    prediction = float(model.predict(example)[0])
    if not np.isfinite(prediction):
        raise ValueError("Model returned a non-finite prediction")
    print(f"Predicted monthly rent: INR {prediction:,.2f}")

if __name__ == "__main__":
    main()
