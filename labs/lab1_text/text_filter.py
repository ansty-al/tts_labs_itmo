import csv
import re
import unicodedata

import os

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


DEV_SET_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "dev_sentences.csv"
)


class TextFilter:

    def __init__(self):

        self.digits_re = re.compile(r"\d")

        self.latin_re = re.compile(r"[A-Za-z]")

        self.tech_symbols_re = re.compile(r"[*/@+<>\\|^~_#={}]")

        self.designations_re = re.compile(r"[%°$€₽№§©®&]")

        self.abbreviations_re = re.compile(
            r"(?<![А-Яа-яЁё])"
            r"(?:"
            r"т\.\s*[дпек]\.|"
            r"др\.|"
            r"им\.|"

            r"мм|см|дм|км|мг|кг|мл|л|мин|сек|"

            r"руб\.|коп\.|тыс\.|млн\.|млрд\.|"

            r"ул\.|пл\.|просп\.|наб\.|пер\.|бул\.|ш\.|"
            r"тр\.|кв\.|обл\.|р-н|пос\.|дер\.|"

            r"рис\.|табл\.|гл\.|стр\.|см\.|ср\.|"
            r"напр\.|прим\.|разд\.|парагр\.|изд\.|ред\.|сост\.|"

            r"проф\.|акад\.|доц\.|ассист\.|инж\.|техн\.|"
            r"дир\.|зав\.|зам\.|тов\.|гр\.|г-н|г-жа|"

            r"чел\.|экз\.|шт\.|ед\."

            r"|гос\.|гос-во|об-во|орг\.|орг-ция|"
            r"предпр\.|учр\.|стр-во|хоз-во"

            r"|г\.|в\.|вв\.|н\.\s*э\.|ч\."
            r")"
            r"(?![А-Яа-яЁё])",
            re.IGNORECASE
        )

        self.initials_re = re.compile(
            r"(?<![А-Яа-яЁё])"
            r"(?:[А-ЯЁ]\.\s*){1,2}[А-ЯЁ][а-яё]+"
            r"|"
            r"[А-ЯЁ][а-яё]+\s+(?:[А-ЯЁ]\.\s*){1,2}"
        )

        self.abbreviation_caps_re = re.compile(
            r"(?<![А-Яа-яЁё])"
            r"[БВГДЖЗЙКЛМНПРСТФХЦЧШЩ]{2,5}"
            r"(?![А-Яа-яЁё])"
        )

        self.quotes_re = re.compile(r"[“”„‟‘’‚]")


        self.repeated_punctuation_re = re.compile(
            r"""
            (?!\?!) [!?]{2,}
            | (?<![.!?]) \.{2} (?!\.)
            | [!?]\.
            | \.[!?]
            """,
            re.VERBOSE,
        )

        self.invisible_re = re.compile(
            r"[\u200B\u200C\u200D\u2060\uFEFF\u00AD\u00A0]"
        )

        self.emoji_re = re.compile(
            r"[\U0001F300-\U0001FAFF"
            r"\U00002700-\U000027BF"
            r"\U0001F1E6-\U0001F1FF]"
        )

        self.text_smiley_re = re.compile(
            r"(?<!\S)"
            r"(?:"
            r"[:;=8][\-^']?[)(DPpOo/\\]"
            r"|"
            r"[)(]{2,}"
            r"|"
            r"\^[_-]\^"
            r"|"
            r"[Tt]_[Tt]"
            r"|"
            r"[Oo]_[Oo]"
            r"|"
            r"[xX][dD]+"
            r")"
            r"(?!\S)"
        )

        self.interjection_re = re.compile(
            r"(?<![А-Яа-яЁё])"
            r"(?:"
            r"ах(?:ах)+|"
            r"ха(?:ха)+|"
            r"хи(?:хи)+|"
            r"уфф+|"
            r"ухх+|"
            r"эхх+|"
            r"ыхх+|"
            r"хм+|"
            r"ыы+|"
            r"м{2,}|"
            r"мгм|"
            r"гм|"
            r"угу|"
            r"бр{2,})",
            re.IGNORECASE
        )

        self.space_before_punctuation_re = re.compile(
            r"\s+([,.!?;:])"
        )

        self.space_after_punctuation_re = re.compile(
            r"([,;:])(?=\S)"
        )

    def filter(self, text):

        if not isinstance(text, str) or not text.strip():
            return 0

        text = unicodedata.normalize("NFC", text)

        checks = [
            self.digits_re,
            self.latin_re,
            self.tech_symbols_re,
            self.designations_re,
            self.abbreviations_re,
            self.initials_re,
            self.abbreviation_caps_re,
            self.quotes_re,
            self.repeated_punctuation_re,
            self.invisible_re,
            self.emoji_re,
            self.text_smiley_re,
            self.interjection_re,
            self.space_before_punctuation_re,
            self.space_after_punctuation_re
        ]

        for pattern in checks:
            if pattern.search(text):
                return 0

        return 1


if __name__ == "__main__":

    textfilter = TextFilter()

    dev_files = pd.read_csv(
        DEV_SET_PATH,
        sep="|",
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        header=0
    )

    dev_files["predicted"] = dev_files["text"].apply(
        textfilter.filter
    )

    prc = precision_score(
        dev_files["is_normalized"],
        dev_files["predicted"]
    )

    rec = recall_score(
        dev_files["is_normalized"],
        dev_files["predicted"]
    )

    f1 = f1_score(
        dev_files["is_normalized"],
        dev_files["predicted"]
    )

    print(
        f"F1 Score is {f1:.4f}, "
        f"Precision is {prc:.4f}, "
        f"Recall is {rec:.4f}"
    )