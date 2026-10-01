import csv
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from praatio import textgrid
import tqdm


RUSLAN_META = '../../data/metadata_RUSLAN_22200_normalized.csv'
ALIGN_DIR = '../../data/RUSLAN_22050_align_v2/'
RESULT_PATH = 'data/RUSLAN_pause_metadata.csv'
FAILED_IDS_PATH = 'data/RUSLAN_pause_failed_ids.txt'  # id файлов, которые не удалось выровнять


# нормализация меток TextGrid
HYPHEN_MAP = str.maketrans({
    '\u2010': '-',  # обычный дефис
    '\u2011': '-',  # неразрывный дефис
})

QUOTE_MAP = str.maketrans({
    '“': '"',
    '”': '"',
    '„': '"',
    '»': '"',
    '«': '"',
    '’': "'",
})

TECH_SYMBOLS_RE = re.compile(r'[*/@+<>\\|^~_#={}%°$€₽№§©®]')

RANDOM_DOT_RE = re.compile(r'(?<=[А-Яа-яЁё])\s*\.(?=\s+[а-яё])')

COMMA_DOT_COMMA_RE = re.compile(r'\.,\.')

PUNCTUATION_SERIES_RE = re.compile(
    r"""
    \?!\.*
    | [!?]{3,}
    | [!?]\.+
    | \.{2,}[!?]
    | \.{2,}
    """,
    re.VERBOSE,
)

SPACE_BEFORE_PUNCTUATION_RE = re.compile(r'\s+([,.!?;:])')

SPACE_AFTER_PUNCTUATION_RE = re.compile(r'([,;:])(?=\S)')

MULTI_SPACE_RE = re.compile(r' {2,}')

# нормализуем метку из word tier: технические символы, дефисы, кавычки, пунктуацию и пробелы
def normalize_textgrid_label(label):

    # пустые метки соответствуют паузам и не требуют нормализации
    if not isinstance(label, str) or not label:
        return label

    text = unicodedata.normalize('NFC', label)

    text = TECH_SYMBOLS_RE.sub(' ', text)

    text = text.translate(HYPHEN_MAP)

    text = text.translate(QUOTE_MAP)

    text = SPACE_BEFORE_PUNCTUATION_RE.sub(r'\1', text)

    def replace_punctuation(match):
        value = match.group(0)

        if '?' in value and '!' in value:
            return '?!'

        if '?' in value:
            return '?'

        if '!' in value:
            return '!'

        return '…'

    text = PUNCTUATION_SERIES_RE.sub(replace_punctuation, text)

    text = COMMA_DOT_COMMA_RE.sub('…', text)

    text = RANDOM_DOT_RE.sub('', text)

    text = SPACE_AFTER_PUNCTUATION_RE.sub(r'\1 ', text)

    text = MULTI_SPACE_RE.sub(' ', text)

    return text.strip()

def safe_normalize_label(label):

    # пустая метка соответствует паузе
    if not isinstance(label, str) or not label.strip():
        return ''

    normalized = normalize_textgrid_label(label)

    # если нормализация удалила всю метку, сохраняем исходное значение
    return normalized if normalized else label.strip()

def read_text_grids(ruslan, align_root, read_phones=False):

    word_docs = []
    phn_docs = []

    phoneme_sequences = []

    # файлы, для которых не удалось прочитать TextGrid
    missing = 0

    for wave_id in tqdm.tqdm(ruslan['id'].values):
        path = os.path.join(align_root, wave_id + '.TextGrid')

        try:
            words = textgrid.openTextgrid(path, True).tiers[0]

        except Exception as e:

            if missing == 0:
                print(f'Не удалось прочитать {os.path.abspath(path)}: {e!r}')

            missing += 1

            continue

        # сохраняем слова, длительности и id файла
        for i in words.entries:
            word_docs.append({
                'label': i.label,
                'duration': i.end - i.start,
                'id': wave_id
            })

    if missing:
        print(f'Не удалось прочитать TextGrid для {missing} из {len(ruslan)} записей метаданных')

    return pd.DataFrame(word_docs, columns=['label', 'duration', 'id'])

# сопоставление токенов TextGrid и исходного текста
WORD_CHAR_RE = re.compile(r'\w')

# Для каждого слова из TextGrid находим соответствующий фрагмент текста и формируем label_raw с пунктуацией
def align_text_and_textgrid(tokens, text, verbose=False):

    tokens = tokens.copy()

    failed_value = np.nan

    def fail(reason, token=None):

        if verbose:
            print(f'Ошибка выравнивания {tokens["id"].iloc[0]}: {reason} (токен: {token!r})')

        tokens['label_raw'] = failed_value

        return tokens

    raw_tokens = []

    text_lower = text.lower()

    previous_word = -1

    # последовательно обрабатываем токены TextGrid
    for t in tokens['label'].values:

        # пустой токен - пауза
        if t == '':
            raw_tokens.append('<SIL>')
            continue

        splits = text_lower.split(t.lower(), maxsplit=1)

        if len(splits) == 1:
            return fail('слово не найдено в тексте', t)

        # между словами допускается только пунктуация
        if WORD_CHAR_RE.search(splits[0]):
            return fail('между словами пропущен текст: ' + splits[0].strip(), t)

        # сохраняем текст вместе с предшествующей пунктуацией
        if previous_word == -1:
            raw_tokens.append(text[:len(splits[0] + t)].strip())

        else:
            # добавляем пунктуацию между предыдущим и текущим словом
            raw_tokens[previous_word] += text[:len(splits[0])].strip()

            # добавляем текущее слово
            raw_tokens.append(text[len(splits[0]):len(splits[0] + t)])

        # оставляем только ещё не сопоставленную часть текста
        text_lower = splits[1]
        text = text[len(splits[0] + t):]

        # запоминаем индекс текущего слова
        previous_word = len(raw_tokens) - 1

    # присоединяем оставшуюся финальную пунктуацию к последнему слову
    if len(text):

        # проверяем, что в TextGrid было хотя бы одно слово
        if previous_word < 0:
            return fail('в TextGrid нет ни одного слова')

        # проверяем отсутствие неразмеченных слов
        if WORD_CHAR_RE.search(text):
            return fail('в конце текста остались неразмеченные слова: ' + text.strip())

        # добавляем финальную пунктуацию к последнему слову
        raw_tokens[previous_word] += text.strip()

    # записываем сопоставленный исходный текст в label_raw
    tokens['label_raw'] = raw_tokens

    return tokens

# формирование меток для предиктора
def add_pause_labels(align):
    
    is_last_word = []
    pause_after = []
    pause_duration = []

    last_word = -1

    for idx, (label, dur) in enumerate(align[['label', 'duration']].values):

        if label == '':

            if last_word >= 0:
                pause_after[last_word] = True
                pause_duration[last_word] = dur

            pause_after.append(False)
            pause_duration.append(0.)
            is_last_word.append(False)

        else:

            pause_after.append(False)
            pause_duration.append(0.)

            is_last_word.append(False)

            last_word = idx

    if last_word >= 0:
        is_last_word[last_word] = True
        
    align = align.copy()

    align['is_last_word'] = is_last_word

    align['is_pause_after'] = pause_after

    align['pause_duration'] = pause_duration

    return align

def main():

    ruslan = pd.read_csv(
        RUSLAN_META,
        sep='|',
        names=['id', 'raw', 'nrm'],
        quoting=csv.QUOTE_NONE
    )

    n_meta = len(ruslan)

    word_df = read_text_grids(ruslan, ALIGN_DIR)

    label_map = {l: safe_normalize_label(l) for l in word_df.label.unique()}

    word_df.label = word_df.label.map(label_map)

    ruslan.nrm = ruslan.nrm.map(
        lambda s:
            unicodedata.normalize('NFC', s)
            if isinstance(s, str)
            else s
    )

    # группируем слова TextGrid по id аудиофайла
    groups = {k: g for k, g in word_df.groupby('id', sort=False)}

    aligns = []
    failed_ids = []
    no_textgrid_ids = []

    for n, i in tqdm.tqdm(ruslan[['nrm', 'id']].values):

        tokens = groups.get(i)

        if tokens is None:
            no_textgrid_ids.append(i)
            continue

        aligned = align_text_and_textgrid(tokens, n)

        if aligned['label_raw'].isna().any():
            failed_ids.append(i)
            continue

        aligns.append(aligned)

    aligns = [add_pause_labels(a) for a in aligns]

    pause_df = pd.concat(aligns)

    pause_df = pause_df[pause_df.label_raw != '<SIL>']

    pause_df = pause_df.reset_index(drop=True)

    pause_df.is_last_word = pause_df.is_last_word.astype(int)
    pause_df.is_pause_after = pause_df.is_pause_after.astype(int)

    pause_df.duration = pause_df.duration.round(6)
    pause_df.pause_duration = pause_df.pause_duration.round(6)

    pause_df['set'] = 'train'

    # файлы с номером, кратным 5, относим к test
    pause_df.loc[pause_df.id.str.split('_', expand=True)[0].astype(int) % 5 == 0, 'set'] = 'test'

    for col in ['label', 'label_raw']:
        pause_df[col] = pause_df[col].str.replace('|', ' ', regex=False)

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    pause_df.to_csv(
        RESULT_PATH,
        sep='|',
        index=False,
        header=True,
        quoting=csv.QUOTE_NONE,
        encoding='utf-8'
    )

    with open(
        FAILED_IDS_PATH,
        'w',
        encoding='utf-8'
    ) as f:
        f.write('\n'.join(failed_ids))

    inner = pause_df[pause_df.is_last_word == 0]

    n_sent = pause_df.id.nunique()

    print('\nСводка\n')
    print(f'Записей в метаданных: {n_meta}\n')
    print(f'Не нашлось в TextGrid: {len(no_textgrid_ids)}\n')
    print(f'Не удалось выровнять с текстом: {len(failed_ids)} (список: {FAILED_IDS_PATH})\n')
    print(f'Успешно обработано: {n_sent}\n')
    print(f'Токенов всего: {len(pause_df)}\n')
    print(f'Доля пауз после токена (без последнего слова): {inner.is_pause_after.mean():.3f}\n')
    print(f'Средняя длительность паузы (без последнего слова): {inner.loc[inner.is_pause_after == 1, "pause_duration"].mean():.3f} с\n')

    # считаем количество предложений в train и test
    sets = pause_df.groupby('set').id.nunique()

    print('Предложений в выборках: ' + ', '.join(f'{k}={v}' for k, v in sets.items()) + '\n')
    print(f'Файл сохранён: {RESULT_PATH}')

if __name__ == '__main__':
    main()