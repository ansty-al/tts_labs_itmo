import csv

import pandas as pd

from text_filter import TextFilter
from text_normalizer import TextNormalizer


INPUT_PATH = "../../data/RUSLAN/metadata_RUSLAN_22200.csv"
OUTPUT_PATH = "../../data/metadata_RUSLAN_22200_normalized.csv"

CSV_KWARGS = {
    "sep": "|",
    "quoting": csv.QUOTE_NONE
}


if __name__ == "__main__":

    text_filter = TextFilter()
    normalizer = TextNormalizer()

    raw = pd.read_csv(
        INPUT_PATH,
        names=["id", "raw"],
        **CSV_KWARGS
    )

    raw["nrm"] = raw["raw"].apply(
        normalizer.normalize
    )

    clean = raw[
        raw["nrm"].apply(text_filter.filter) == 1
    ]

    print(
        f"Num rows before cleaning: {len(raw)}; "
        f"after cleaning: {len(clean)}"
    )

    clean[["id", "raw", "nrm"]].to_csv(
        OUTPUT_PATH,
        index=False,
        header=False,
        **CSV_KWARGS
    )