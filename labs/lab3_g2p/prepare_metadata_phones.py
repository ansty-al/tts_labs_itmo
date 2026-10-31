import os
import csv
import pandas as pd

from g2p import G2P, load_phones

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(LAB_DIR, '..', '..'))

TOKENS_PATH = os.path.join(LAB_DIR, 'data', 'tokens_phones_sil.csv')
LEXICON_PATH = os.path.join(LAB_DIR, 'data', 'lexicon.csv')
MFA_DICT_PATH = os.path.join(LAB_DIR, 'data', 'russian_mfa.dict')
PHONES_PATH = os.path.join(PROJECT_DIR, 'common', 'phones_mfa.txt')
METADATA_PATH = os.path.join(PROJECT_DIR, 'data', 'metadata_RUSLAN_22200_normalized.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'data')

metadata = pd.read_csv(METADATA_PATH, sep='|', header=None, dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
tokens = pd.read_csv(TOKENS_PATH, sep='|', dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
tokens['token_id'] = pd.to_numeric(tokens['token_id'])
tokens = tokens.sort_values(['id', 'token_id'], kind='stable')
phones = load_phones(PHONES_PATH)

model = G2P.from_files(lexicon_path=LEXICON_PATH, mfa_path=MFA_DICT_PATH, phones_path=PHONES_PATH)

# собираем эталонные и предсказанные транскрипции
gt_rows = []
pred_rows = []

for _, row in metadata.iterrows():
    utterance_id = row[0]
    normalized_text = row[2]

    group = tokens[tokens['id'] == utterance_id]
    if group.empty:
        continue

    word_tokens = group['token'].tolist()

    gt = [phone for transcription in group['phones'] for phone in transcription.split()]
    pred = [phone for pronunciation in model.transcribe(word_tokens) for phone in pronunciation]
    
    gt_rows.append([utterance_id, normalized_text, ' '.join(gt)])
    pred_rows.append([utterance_id, normalized_text, ' '.join(pred)])


# проверяем, что все фонемы входят в инвентарь
for rows in (gt_rows, pred_rows):
    unknown = {
        phone
        for _, _, transcription in rows
        for phone in transcription.split()
        if phone not in phones
    }
    if unknown:
        raise ValueError(f'Фонемы отсутствуют в phones_mfa.txt: {sorted(unknown)}')

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename, rows in [('metadata_phones_gt.csv', gt_rows), ('metadata_phones.csv', pred_rows)]:
    with open(os.path.join(OUTPUT_DIR, filename), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(
            f,
            delimiter='|',
            quoting=csv.QUOTE_NONE,
            escapechar='\\',
            lineterminator='\n'
        )
        writer.writerows(rows)

    print(f'Сохранен {filename}: {len(rows)} предложений')
