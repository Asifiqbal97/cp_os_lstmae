"""Generates synthetic data matching the real 46-feature schema, with
distinct-ish distributions per family, purely to smoke-test the pipeline
end-to-end before real data is available. NOT for reporting real metrics.
"""

import numpy as np
import pandas as pd

from src.data.loader import FEATURE_COLUMNS

RNG = np.random.default_rng(42)

FAMILY_SUBTYPES = {
    "Benign": ["Benign"],
    "Spoofing": ["ARP_Spoofing"],
    "MQTT": ["MQTT-DDoS-Connect_Flood"],
    "Recon": ["Recon-Port_Scan"],
    "DDoS": ["TCP_IP-DDoS-SYN1"],
    "DoS": ["TCP_IP-DoS-SYN1"],
}


def make_family_block(family, n, shift):
    X = RNG.normal(loc=shift, scale=1.0, size=(n, len(FEATURE_COLUMNS)))
    X = np.abs(X)  # feature values are non-negative in the real data
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    subtype = FAMILY_SUBTYPES[family][0]
    df["Attack_type"] = subtype
    df["Attack_label"] = 0 if family == "Benign" else 1
    return df


def make_synthetic_dataset(n_per_family=400):
    shifts = {"Benign": 0.0, "Spoofing": 1.0, "MQTT": 2.0, "Recon": 3.0, "DDoS": 4.0, "DoS": 5.0}
    blocks = [make_family_block(fam, n_per_family, shift) for fam, shift in shifts.items()]
    return pd.concat(blocks, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)


if __name__ == "__main__":
    n_per_family = 300
    train = make_synthetic_dataset(n_per_family)
    test = make_synthetic_dataset(max(n_per_family // 4, 50))
    train.to_csv("data/raw/_synthetic_train.csv", index=False)
    test.to_csv("data/raw/_synthetic_test.csv", index=False)

    # Raw, unshuffled Benign-only files (native capture order), for Gate 2
    # sequence building -- kept separate from the shuffled merged files above.
    benign_train = make_family_block("Benign", n_per_family * 3, 0.0)
    benign_test = make_family_block("Benign", n_per_family, 0.0)
    benign_train.to_csv("data/raw/_synthetic_benign_train.csv", index=False)
    benign_test.to_csv("data/raw/_synthetic_benign_test.csv", index=False)

    print("Wrote synthetic train/test/benign CSVs for smoke testing.")
