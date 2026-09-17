import re
import unicodedata

class TextNormalizer:

    def __init__(self):

        self.multi_space_re = re.compile(r" {2,}")

        # дефисы
        self.hyphen_map = str.maketrans({"-": "-"})

        # технические символы
        self.tech_symbols_re = re.compile(
            r"[*/@+<>\\|^~_#={}%°$€₽№§©®]"
        )

        # кавычки
        self.quote_map = str.maketrans({
            "“": '"',
            "”": '"',
            "„": '"',
            "»": '"',
            "«": '"',
            "’": "'"
        })

        # случайные точки внутри предложения
        self.random_dot_re = re.compile(
            r"(?<=[А-Яа-яЁё])\s*\.(?=\s+[а-яё])"
        )

        # .,. → …
        self.comma_dot_comma_re = re.compile(r"\.,\.")

        # серии знаков препинания
        self.punctuation_series_re = re.compile(
            r"""
            \?!\.*
            | [!?]{3,}
            | [!?]\.+
            | \.{2,}[!?]
            | \.{2,}
            """,
            re.VERBOSE,
        )

        # пробелы вокруг пунктуации
        self.space_before_punctuation_re = re.compile(
            r"\s+([,.!?;:])"
        )

        self.space_after_punctuation_re = re.compile(
            r"([,;:])(?=\S)"
        )

        # однозначные общеупотребительные сокращения
        self.abbreviations = {
            "т.е.": "то есть",
            "т. е.": "то есть",
            "т.к.": "так как",
            "т. к.": "так как",
            "т.д.": "так далее",
            "т. д.": "так далее",
            "т.п.": "тому подобное",
            "т. п.": "тому подобное",
        }

        self.abbreviation_re = re.compile(
            r"(?<![А-Яа-яЁё])"
            r"(т\.\s*е\.|т\.\s*к\.|т\.\s*д\.|т\.\s*п\.)"
            r"(?![А-Яа-яЁё])",
            re.IGNORECASE
        )

        # числа в буквенно-численных обозначениях
        self.alphanumeric_number_re = re.compile(
            r"\b([А-ЯЁа-яё]+)-(\d+)\b"
        )

    # расшифровка цифр
    def number_to_words(self, number):

        number = int(number)

        units = {
            0: "ноль",
            1: "один",
            2: "два",
            3: "три",
            4: "четыре",
            5: "пять",
            6: "шесть",
            7: "семь",
            8: "восемь",
            9: "девять",
        }

        teens = {
            10: "десять",
            11: "одиннадцать",
            12: "двенадцать",
            13: "тринадцать",
            14: "четырнадцать",
            15: "пятнадцать",
            16: "шестнадцать",
            17: "семнадцать",
            18: "восемнадцать",
            19: "девятнадцать",
        }

        tens = {
            20: "двадцать",
            30: "тридцать",
            40: "сорок",
            50: "пятьдесят",
            60: "шестьдесят",
            70: "семьдесят",
            80: "восемьдесят",
            90: "девяносто",
        }

        hundreds = {
            100: "сто",
            200: "двести",
            300: "триста",
            400: "четыреста",
            500: "пятьсот",
        }

        if number < 10:
            return units[number]

        if number < 20:
            return teens[number]

        if number < 100:
            return tens[number // 10 * 10] + (
                " " + units[number % 10]
                if number % 10
                else ""
            )

        if number <= 500:
            hundred = number // 100 * 100
            remainder = number % 100

            if remainder == 0:
                return hundreds[hundred]

            return hundreds[hundred] + " " + self.number_to_words(remainder)

        return str(number)

    def normalize(self, text):

        if not isinstance(text, str) or not text:
            return ""

        text = unicodedata.normalize("NFC", text)

        text = text.translate(self.hyphen_map)

        def replace_alphanumeric_number(match):
            letters = match.group(1)
            number = match.group(2)

            return letters + " " + self.number_to_words(number)

        text = self.alphanumeric_number_re.sub(
            replace_alphanumeric_number,
            text
        )

        def replace_abbreviation(match):
            abbreviation = match.group(0).lower()
            abbreviation = abbreviation.replace(" ", "")

            return self.abbreviations[abbreviation]

        def replace_punctuation(match):
            value = match.group(0)

            if "?" in value and "!" in value:
                return "?!"

            if "?" in value:
                return "?"

            if "!" in value:
                return "!"

            return "…"

        text = self.abbreviation_re.sub(
            replace_abbreviation,
            text
        )

        text = text.translate(self.quote_map)

        text = self.tech_symbols_re.sub(" ", text)

        text = self.space_before_punctuation_re.sub(
            r"\1",
            text
        )

        text = self.punctuation_series_re.sub(
            replace_punctuation,
            text
        )

        text = self.comma_dot_comma_re.sub(
            "…",
            text
        )

        text = self.random_dot_re.sub(
            "",
            text
        )

        text = self.space_after_punctuation_re.sub(
            r"\1 ",
            text
        )

        text = self.multi_space_re.sub(
            " ",
            text
        )

        text = unicodedata.normalize("NFC", text)

        return text.strip()