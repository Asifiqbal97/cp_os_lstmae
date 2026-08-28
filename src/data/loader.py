"""Loads CICIoMT2024 merged CSVs and attaches a `family` column.

Kept deliberately dumb: no filtering, no splitting -- that's splitter.py's
job. This module only knows how to read the file and label it correctly.
"""

import pandas as pd

from .family_map import subtype_to_family

FEATURE_COLUMNS = [
    "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]
LABEL_COLUMNS = ["Attack_type", "Attack_label"]
ALL_COLUMNS = FEATURE_COLUMNS + LABEL_COLUMNS


def load_csv(path: str) -> pd.DataFrame:
    """Load a CICIoMT2024 CSV and add a `family` column derived from
    Attack_type. Validates the schema matches what we expect -- fails loudly
    on mismatch rather than silently dropping/misaligning columns.

    Loads features as float32 (half the memory of pandas' float64 default)
    and Attack_type as category -- matters on a ~10GB RAM machine with a
    multi-GB CSV.
    """
    dtype_map = {col: "float32" for col in FEATURE_COLUMNS}
    dtype_map["Attack_type"] = "category"
    dtype_map["Attack_label"] = "int8"

    df = pd.read_csv(path, dtype=dtype_map)

    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns: {missing}")

    df["family"] = df["Attack_type"].apply(subtype_to_family).astype("category")
    return df


if __name__ == "__main__":
    # Sanity check against the sample file, before real merged data arrives.
    import sys
    sample_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/user-data/uploads/ARP_Spoofing_test_labelled.csv"
    df = load_csv(sample_path)
    print(df[["Attack_type", "family", "Attack_label"]].value_counts())
    print(f"\nShape: {df.shape}")
