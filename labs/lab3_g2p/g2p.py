'''
Порядок принятия решения:
1) произношение из lexicon.csv;
2) если там нет - произношение из russian_mfa.dict;
3) нет ни в одном словаре - правила RuleG2P.

В данном случае:
OOV - слова нет в lexicon.csv, но может быть в russian_mfa.dict.
Residual - слова нет в обоих словарях.
'''

import argparse
import copy
import csv
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tqdm

TOKENS_PHONES_DATA = 'data/tokens_phones.csv'
LEXICON_DATA = 'data/lexicon.csv'
MFA_DICT_DATA = 'data/russian_mfa.dict'
PHONES_DATA = 'data/phones_mfa.txt'

SIL = '<SIL>'
LENGTH_MARK = 'ː'
DENTAL = '\u032a'  # обозначение дентальности, как в t̪ d̪ s̪ z̪ n̪

# соответствия русских букв фонемам в обозначениях MFA
HARD = {'б': 'b', 'в': 'v', 'г': '\u0261', 'д': 'd' + DENTAL, 'з': 'z' + DENTAL, 'к': 'k',
        'л': 'ɫ', 'м': 'm', 'н': 'n' + DENTAL, 'п': 'p', 'р': 'r', 'с': 's' + DENTAL,
        'т': 't' + DENTAL, 'ф': 'f', 'х': 'x', 'ц': 't' + DENTAL + 's' + DENTAL,
        'ж': 'ʐ', 'ш': 'ʂ'}  # твердые

SOFT = {'б': 'bʲ', 'в': 'vʲ', 'г': 'ɟ', 'д': 'dʲ', 'з': 'zʲ', 'к': 'c', 'л': 'ʎ', 'м': 'mʲ',
        'н': 'ɲ', 'п': 'pʲ', 'р': 'rʲ', 'с': 'sʲ', 'т': 'tʲ', 'ф': 'fʲ', 'х': 'ç'}  # мягкие

ALWAYS_SOFT = {'ч': 'tɕ', 'щ': 'ɕ'}
HARD_ONLY = set('жшц')                 
VOWEL_LETTERS = 'аеёиоуыэюя'
SOFTENING = set('яёюие') # гласные, смягчающие предшествующий согласный             
IOTATED = set('яёюе')                 

VOWEL_PHONES = set('a æ e i o u ɐ ə ɛ ɨ ɪ ɵ ʉ ʊ'.split())
REDUCED = set('ɐ ə ɪ ʊ'.split())  # только в безударных слогах     
AMBIGUOUS = set('ɨ ʉ ɛ'.split())  # могут быть и в ударных, и в безударных слогах

# пары оглушения/озвончения
DEVOICE = {'б': 'п', 'в': 'ф', 'г': 'к', 'д': 'т', 'з': 'с', 'ж': 'ш'}
VOICE = {v: k for k, v in DEVOICE.items()}
VOICELESS_TRIGGER = set('пфктсшхцчщ') 
VOICED_TRIGGER = set('бгджз')         

RULES = ('palatalization', 'voicing', 'spelling', 'reduction')
SPELLING = (('сч', 'щ'), ('зч', 'щ'), ('тч', 'ч'), ('дч', 'ч'),
            ('вств', 'ств'), ('стн', 'сн'), ('здн', 'зн'), ('стл', 'сл'), ('лнц', 'нц'), ('рдц', 'рц'))

DICT_WORD = re.compile(r'^[а-яё]+(?:-[а-яё]+)*$')

def normalize_token(token) -> str:
    return unicodedata.normalize('NFC', str(token)).lower() 

# подсчет гласных для подсчета слогов
def count_vowel_letters(word: str) -> int:
    return sum(ch in VOWEL_LETTERS for ch in word)

# запасной вариант ударения - на предпоследний слог
def penultimate_stress(word: str) -> int:
    return max(count_vowel_letters(word) - 2, 0)

# пытаемся восстановить ударение по транскрипции и редукции
def stressed_syllable(word: str, phones: str) -> int | None:
    vowels = [p for p in phones.split() if p in VOWEL_PHONES]
    if len(vowels) < 2 or len(vowels) != count_vowel_letters(word):
        return None
    full = [i for i, p in enumerate(vowels) if p not in REDUCED]
    if len(full) > 1:
        full = [i for i in full if vowels[i] not in AMBIGUOUS]
    return full[0] if len(full) == 1 else None

def to_target(pron: str) -> str:
    return ' '.join(phone.replace('ː', '') for phone in pron.split())

def load_phones(path: str = PHONES_DATA) -> set[str]:
    with open(path, encoding='utf-8') as fh:
        return {unicodedata.normalize('NFC', phone) for phone in fh.read().split()}

def load_lexicon(path: str = LEXICON_DATA) -> dict[str, Counter]:
    table = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE, keep_default_na=False)
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for token, phones, frequency in table[['token', 'phones', 'frequency']].values:
        lexicon[unicodedata.normalize('NFC', str(token))][phones] += int(frequency)
    return lexicon

def load_mfa_dict(path: str = MFA_DICT_DATA, phones: set[str] | None = None) -> dict[str, Counter]:

    dictionary: dict[str, Counter] = defaultdict(Counter)
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            if '\t' in line:
                fields = line.rstrip().split('\t')
                word, pron = fields[0], fields[-1]
                try:
                    weight = float(fields[1]) if len(fields) > 2 else 1.0
                except ValueError:
                    weight = 1.0
            else:
                word, *rest = line.split()
                pron, weight = ' '.join(rest), 1.0
            word = unicodedata.normalize('NFC', word).lower()
            if not DICT_WORD.match(word) or 'spn' in pron.split():
                continue
            mapped = to_target(pron)
            if mapped:
                dictionary[word][mapped] += weight
    return dict(dictionary)

# пытаемся статистически предсказать, какое ударение чаще встречается у слов с похожими окончаниями
class StressPredictor:

    MAX_SUFFIX = 5
    MIN_COUNT = 5
    
    def __init__(self, pronunciations: dict[str, str]) -> None:
        self.counts: dict[str, Counter] = defaultdict(Counter)
        for word, phones in pronunciations.items():
            position = stressed_syllable(word, phones)
            if position is None:
                continue
            offset = position - count_vowel_letters(word)  # -1 - последний слог
            for length in range(min(self.MAX_SUFFIX, len(word)) + 1):
                self.counts[word[len(word) - length:]][offset] += 1

    def __call__(self, word: str) -> int:
        n = count_vowel_letters(word)
        for length in range(min(self.MAX_SUFFIX, len(word)), -1, -1):
            stats = self.counts.get(word[len(word) - length:])
            if not stats or sum(stats.values()) < self.MIN_COUNT:
                continue
            for offset, _ in stats.most_common():
                if n + offset >= 0:
                    return n + offset      
        return penultimate_stress(word)

@dataclass
class Unit:
    kind: str  # C - согласная, J - звук [j], V - гласная
    letter: str = ''
    soft: bool = False

# правила для слов, которых нет в обоих наших словарях
class RuleG2P:
    def __init__(self, stress=None, rules=RULES) -> None:
        self.stress = stress
        self.rules = frozenset(rules)

    @staticmethod
    # фонемы, которые может выдать модель (чтобы проверить инвентарь)
    def emitted_phones() -> set[str]:
        phones = set(HARD.values()) | set(SOFT.values()) | set(ALWAYS_SOFT.values())
        return {unicodedata.normalize('NFC', p) for p in phones | VOWEL_PHONES | {'j'}}

    def __call__(self, word: str, stress: int | None = None) -> list[str]:
        if '-' in word:
            return [p for part in word.split('-') if part for p in self(part)]
        if stress is None:
            stress = self._stressed_vowel(word)
        units = self._parse(self._spell(word))
        if 'voicing' in self.rules:
            self._assimilate(units)
        phones, vowel_index = [], -1
        for k, unit in enumerate(units):
            if unit.kind == 'C':
                phones.append(ALWAYS_SOFT[unit.letter] if unit.letter in ALWAYS_SOFT
                              else (SOFT if unit.soft else HARD)[unit.letter])
            elif unit.kind == 'J':
                phones.append('j')
            else:
                vowel_index += 1
                phones.append(self._vowel(units, k, vowel_index, stress))
        return phones

    # выбор ударной гласной
    def _stressed_vowel(self, word: str) -> int | None:
        if 'reduction' not in self.rules:
            return None
        vowels = [ch for ch in word if ch in VOWEL_LETTERS]
        if 'ё' in vowels:
            return vowels.index('ё')
        if len(vowels) == 1:
            return 0
        if len(vowels) > 1 and self.stress is not None:
            return self.stress(word)
        return None

    # несколько правил чтения (довольно частые упрощения, ради эксперимента)
    def _spell(self, word: str) -> str:
        if 'spelling' not in self.rules:
            return word
        for old, new in SPELLING:
            word = word.replace(old, new)
        if word.endswith(('ого', 'его')):
            word = word[:-2] + 'во'
        if word.endswith('ться'):
            word = word[:-4] + 'ца'
        elif word.endswith('тся'):
            word = word[:-3] + 'ца'    
        return word

    # преобразование в Unit (выше) с учетом мягкости, "ь" и "ъ", йотирования
    def _parse(self, word: str) -> list[Unit]:
        palatal = 'palatalization' in self.rules
        units: list[Unit] = []
        for i, ch in enumerate(word):
            prev = word[i - 1] if i else ''
            nxt = word[i + 1] if i + 1 < len(word) else ''
            if ch in VOWEL_LETTERS:
                after_sign = prev in ('ь', 'ъ')
                if (ch in IOTATED and (not palatal or i == 0 or prev in VOWEL_LETTERS or after_sign)) \
                        or (ch == 'и' and after_sign):
                    units.append(Unit('J'))
                units.append(Unit('V', ch))
            elif ch == 'ь':
                if palatal and units and units[-1].kind == 'C' and units[-1].letter not in HARD_ONLY:
                    units[-1].soft = True
            elif ch == 'й':
                units.append(Unit('J'))
            elif ch in HARD or ch in ALWAYS_SOFT:
                soft = palatal and (ch in ALWAYS_SOFT or
                                    (ch not in HARD_ONLY and (nxt in SOFTENING or nxt == 'ь')))
                units.append(Unit('C', ch, soft))
        return units

    @staticmethod
    # звонкость/глухость
    def _assimilate(units: list[Unit]) -> None:
        for k in range(len(units) - 1, -1, -1):
            unit = units[k]
            if unit.kind != 'C' or not (unit.letter in DEVOICE or unit.letter in VOICE):
                continue
            nxt = units[k + 1] if k + 1 < len(units) else None
            if nxt is None or (nxt.kind == 'C' and nxt.letter in VOICELESS_TRIGGER):
                unit.letter = DEVOICE.get(unit.letter, unit.letter)
            elif nxt.kind == 'C' and nxt.letter in VOICED_TRIGGER:
                unit.letter = VOICE.get(unit.letter, unit.letter)

    # выбор фонемы для гласной с учетом ударения и соседних звуков
    def _vowel(self, units: list[Unit], k: int, vowel_index: int, stress: int | None) -> str:
        unit = units[k]
        prev = units[k - 1] if k else None
        nxt = units[k + 1] if k + 1 < len(units) else None
        prev_soft = prev is not None and (prev.kind == 'J' or (prev.kind == 'C' and prev.soft))
        next_soft = nxt is not None and (nxt.kind == 'J' or (nxt.kind == 'C' and nxt.soft))
        prev_hush = prev is not None and prev.kind == 'C' and prev.letter in HARD_ONLY

        if 'reduction' not in self.rules or stress is None or vowel_index == stress:
            return self._stressed(unit.letter, prev_soft, next_soft, prev_hush)
        pretonic = vowel_index < stress
        first_pretonic = pretonic and stress - vowel_index == 1
        return self._unstressed(unit.letter, prev, pretonic, first_pretonic or k == 0, prev_hush)

    @staticmethod
    # соответствия для ударных
    def _stressed(ch: str, prev_soft: bool, next_soft: bool, prev_hush: bool) -> str:
        if ch in 'ая':
            return 'æ' if prev_soft and next_soft else 'a'
        if ch in 'оё':
            return 'o' if prev_hush else ('ɵ' if prev_soft else 'o')
        if ch == 'у':
            return 'ʉ' if prev_soft and next_soft else 'u'
        if ch == 'ю':
            return 'ʉ' if next_soft else 'u'
        if ch == 'э':
            return 'ɛ'
        if ch == 'ы':
            return 'ɨ'
        if ch == 'е':
            return 'ɛ' if prev_hush else 'e'
        return 'ɨ' if prev_hush else 'i'  # 'и'

    @staticmethod
    # редукция безударных
    def _unstressed(ch: str, prev: Unit | None, pretonic: bool, near_stress: bool,
                    prev_hush: bool) -> str:
        if ch in 'ао':
            if ch == 'а' and prev is not None and prev.kind == 'C' and prev.letter in 'чщ':
                return 'ɪ' if pretonic else 'ə'
            return 'ɐ' if near_stress else 'ə'
        if ch == 'я':
            return 'ɪ' if pretonic else 'ə'
        if ch == 'е':
            return ('ɨ' if pretonic else 'ə') if prev_hush else 'ɪ'
        if ch == 'и':
            return 'ɨ' if prev_hush else 'ɪ'
        return {'ы': 'ɨ', 'э': 'ɛ', 'у': 'ʊ', 'ю': 'ʉ', 'ё': 'ɵ'}[ch]

class G2P:
    def __init__(self, lexicon: dict[str, Counter] | None = None,
                 mfa_dict: dict[str, Counter] | None = None,
                 phones: set[str] | None = None,
                 rules=RULES) -> None:
        
        self.lexicon = lexicon or {}
        self.phones = phones
        self.pron: dict[str, str] = {}
        for source in (mfa_dict or {}, self.lexicon):
            for word, variants in source.items():
                if variants:
                    self.pron[word] = variants.most_common(1)[0][0]
        self.stress = StressPredictor(self.pron)
        self.oov = RuleG2P(self.stress, rules)
        self._cache: dict[str, list[str]] = {}
        if phones is not None:
            missing = RuleG2P.emitted_phones() - phones
            if missing:
                raise ValueError(f'the rules emit phones outside the target list: {sorted(missing)}')

    @classmethod
    def from_files(cls, lexicon_path: str = LEXICON_DATA, mfa_path: str = MFA_DICT_DATA,
                   phones_path: str | None = None, rules=RULES) -> 'G2P':
        phones = load_phones(phones_path)
        return cls(load_lexicon(lexicon_path), load_mfa_dict(mfa_path, phones), phones, rules)

    def with_rules(self, rules, stress=None) -> 'G2P':
        clone = copy.copy(self)
        clone.oov = RuleG2P(stress or self.stress, rules)
        clone._cache = {}
        return clone

    def is_known(self, token) -> bool:
        word = normalize_token(token)
        if word in self.pron:
            return True
        return bool(word) and '-' in word and all(self.is_known(part) for part in word.split('-'))

    def transcribe(self, tokens: list[str] | np.ndarray) -> list[list[str]]:
        return [[SIL] if token == SIL else self.transcribe_word(token) for token in tokens]

    def transcribe_word(self, token) -> list[str]:
        word = normalize_token(token)
        if word not in self._cache:
            self._cache[word] = self._word(word)
        return list(self._cache[word])

    def _word(self, word: str) -> list[str]:
        if word in self.pron:
            return self.pron[word].split()
        if '-' in word:
            return [p for part in word.split('-') if part for p in self._word(part)]
        return self.oov(word)

# подсчет метрик
def phone_error_rate(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    total_distance = 0
    total_length = 0
    for ref, hyp in zip(reference, hypothesis):
        total_length += len(ref)
        if not hyp:
            total_distance += len(ref)
            continue
        previous = list(range(len(hyp) + 1))
        for i, ref_phone in enumerate(ref, start=1):
            current = [i] + [0] * len(hyp)
            for j, hyp_phone in enumerate(hyp, start=1):
                current[j] = min(previous[j] + 1,
                                 current[j - 1] + 1,
                                 previous[j - 1] + (ref_phone != hyp_phone))
            previous = current
        total_distance += previous[-1]
    return total_distance / total_length if total_length else float('nan')

def word_accuracy(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    if not reference:
        return float('nan')
    return float(np.mean([ref == hyp for ref, hyp in zip(reference, hypothesis)]))

# токены и эталонные транскрипции, строки без транскрипции или с spn исключаются
def load_tokens(path: str = TOKENS_PHONES_DATA) -> pd.DataFrame:
    tokens = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    tokens['token'] = tokens.token.astype(str).map(lambda s: unicodedata.normalize('NFC', s))
    tokens['phones'] = tokens.phones.fillna('').astype(str)
    
    unpronounced = (tokens.phones == '') | tokens.phones.str.split().map(lambda p: 'spn' in p)
    print(f'dropping {unpronounced.sum()} tokens without a transcription '
          f'({unpronounced.mean():.2%})')
    return tokens[~unpronounced].reset_index(drop=True)

# частоты вариантов произношения
def build_lexicon(tokens: pd.DataFrame) -> dict[str, Counter]:
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for token, phones in tokens[['token', 'phones']].values:
        lexicon[token][phones] += 1  
    return lexicon

# общая оценка
def calc_metrics(g2p: G2P, tokens: pd.DataFrame, lexicon: dict[str, Counter],
                 verbose: bool = True) -> dict[str, float]:
    sentence_ref: list[list[str]] = []
    sentence_hyp: list[list[str]] = []
    oov_ref: list[list[str]] = []
    oov_hyp: list[list[str]] = []
    in_vocab_ref: list[list[str]] = []
    in_vocab_hyp: list[list[str]] = []
    residual_ref: list[list[str]] = []
    residual_hyp: list[list[str]] = []

    for _, sentence in tqdm.tqdm(tokens.groupby('id', sort=False), disable=not verbose):
        sentence = sentence.sort_values('token_id')
        words = sentence.token.values
        predicted = g2p.transcribe(words)
        
        if len(predicted) != len(words):
            raise ValueError(f'transcribe returned {len(predicted)} outputs '
                             f'for {len(words)} tokens')
        reference = [p.split() for p in sentence.phones.values]

        sentence_ref.append([p for word in reference for p in word if p != SIL])
        sentence_hyp.append([p for word in predicted for p in word if p != SIL])
        
        for token, ref, hyp in zip(words, reference, predicted):
            if token in lexicon:
                in_vocab_ref.append(ref)
                in_vocab_hyp.append(hyp)
            else:
                oov_ref.append(ref)
                oov_hyp.append(hyp)
                if not g2p.is_known(token):
                    residual_ref.append(ref)
                    residual_hyp.append(hyp)

    n_words = len(in_vocab_ref) + len(oov_ref)
    metrics = {
        'running': phone_error_rate(sentence_ref, sentence_hyp),
        'in_vocab': phone_error_rate(in_vocab_ref, in_vocab_hyp),
        'oov': phone_error_rate(oov_ref, oov_hyp),
        'residual': phone_error_rate(residual_ref, residual_hyp),
        'residual_words': len(residual_ref),
    }
    if verbose:
        print(f'running text  PER {metrics["running"]:.4f}')
        print(f'in-vocabulary PER {metrics["in_vocab"]:.4f}  '
              f'({len(in_vocab_ref)} words, {len(in_vocab_ref) / n_words:.1%})')
        print(f'OOV           PER {metrics["oov"]:.4f}  '
              f'word accuracy {word_accuracy(oov_ref, oov_hyp):.4f}  '
              f'({len(oov_ref)} words, {len(oov_ref) / n_words:.1%})')
        print(f'residual      PER {metrics["residual"]:.4f}  '
              f'word accuracy {word_accuracy(residual_ref, residual_hyp):.4f}  '
              f'({len(residual_ref)} words, {len(residual_ref) / n_words:.1%})  '
              f'[in neither dictionary]')
    return metrics

# сравнение действенности правил
def ablation(g2p: G2P, train: pd.DataFrame, test: pd.DataFrame, lexicon: dict[str, Counter]) -> None:

    configs = [
        ('naive letter -> phone map', (), None),
        ('+ palatalization, iotation', ('palatalization',), None),
        ('+ voicing assimilation, devoicing', ('palatalization', 'voicing'), None),
        ('+ spelling rules', ('palatalization', 'voicing', 'spelling'), None),
        ('+ reduction, stress on penultimate', RULES, penultimate_stress),
        ('+ reduction, learned stress', RULES, None),
    ]
    train_words = [normalize_token(t) for t in train.token.values]
    train_refs = [p.split() for p in train.phones.values]
    residual = test[~test.token.map(g2p.is_known)]
    residual_words = [normalize_token(t) for t in residual.token.values]
    residual_refs = [p.split() for p in residual.phones.values]

    rows = []
    for name, rules, stress in tqdm.tqdm(configs):
        variant = g2p.with_rules(rules, stress)
        cache: dict[str, list[str]] = {}

        def rule_per(words, refs):
            for word in set(words) - set(cache):
                cache[word] = variant.oov(word)
            return phone_error_rate(refs, [cache[w] for w in words])

        rows.append({
            'configuration': name,
            'train PER': rule_per(train_words, train_refs),
            'test residual PER': rule_per(residual_words, residual_refs),
            'test running text PER': calc_metrics(variant, test, lexicon, verbose=False)['running'],
        })

    print(f'\nresidual test tokens: {len(residual_words)}')
    print(pd.DataFrame(rows).to_string(index=False, float_format='{:.4f}'.format))

# слова, отсутствовавшие в обоих словарях, где больше всего ошибок
def show_errors(g2p: G2P, test: pd.DataFrame, n: int = 40) -> None:

    residual = test[~test.token.map(g2p.is_known)].drop_duplicates('token')
    rows = []
    for token, phones in residual[['token', 'phones']].values:
        ref, hyp = phones.split(), g2p.transcribe_word(token)
        errors = phone_error_rate([ref], [hyp]) * len(ref)
        rows.append((errors, token, phones, ' '.join(hyp)))
    print(f'residual types: {len(rows)}; first {n} by number of errors')
    for errors, token, ref, hyp in sorted(rows, reverse=True)[:n]:
        
        print(f'{token:<22}{errors:>3.0f}  ref: {ref}\n{"":<25}hyp: {hyp}')

# главный эксперимент
def test_g2p(run_ablation: bool = False, run_errors: bool = False) -> None:

    tokens = load_tokens()
    train = tokens[tokens.set == 'train']
    test = tokens[tokens.set == 'test']
    print(f'train {len(train)} tokens / test {len(test)} tokens')

    lexicon = build_lexicon(train)
    print(f'lexicon: {len(lexicon)} types, '
          f'{sum(1 for c in lexicon.values() if len(c) > 1)} with several pronunciations')

    phones = load_phones()
    mfa_dict = load_mfa_dict(phones=phones)
    print(f'russian_mfa.dict: {len(mfa_dict)} words')

    g2p = G2P(lexicon, mfa_dict, phones)

    print('\nCalculate metrics, training fold:')
    calc_metrics(g2p, train, lexicon)

    print('\nCalculate metrics, testing fold:')
    calc_metrics(g2p, test, lexicon)

    if run_ablation:
        print('\nAblation of the rules:')
        ablation(g2p, train, test, lexicon)

    if run_errors:
        print('\nWorst residual tokens, test fold:')
        show_errors(g2p, test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score the G2P module on the prepared data.')
    parser.add_argument('--ablation', action='store_true', help='also measure every rule')
    parser.add_argument('--errors', action='store_true', help='also list the worst residual tokens')
    args = parser.parse_args()
    test_g2p(args.ablation, args.errors)
