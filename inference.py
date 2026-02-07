import pickle
import pandas as pd
from pathlib import Path

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def main():
    model_path = Path("artifacts/best_model.pkl")
    encoder_path = Path("artifacts/y_encoder.pkl")
    data_path = Path("data/raw/test.csv")
    output_path = Path("predictions/preds.csv")

    output_path.parent.mkdir(exist_ok=True)

    # load artifacts
    model = load_pickle(model_path)
    encoder = load_pickle(encoder_path)

    # load data
    df = pd.read_csv(data_path)

    # predict
    y_pred_encoded = model.predict(df)

    # decode labels
    y_pred = encoder.inverse_transform(y_pred_encoded)

    # save results
    out = pd.DataFrame({
        "id": df["id"] if "id" in df.columns else range(len(df)),
        "Heart Disease": y_pred_encoded,
    })
    
    out.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
