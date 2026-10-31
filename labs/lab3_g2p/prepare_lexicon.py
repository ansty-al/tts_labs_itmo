import csv
import glob
import os
import pandas as pd
import unicodedata
from collections import Counter, defaultdict

from praatio import textgrid
import tqdm

ALIGN_DIR = '../../data/RUSLAN_22050_align_v2/'
NORMALIZED_META = '../../data/metadata_RUSLAN_22200_normalized.csv'
TOKENS_PATH = 'data/tokens_phones.csv'
TOKENS_SIL_PATH = 'data/tokens_phones_sil.csv'
LEXICON_PATH = 'data/lexicon.csv'

SIL = '<SIL>'
LENGTH_MARK = 'ː'
TEST_EVERY = 5  # lab 2's split rule: test iff the utterance number divides by 5


def phones_of_token(phones: list[tuple[float, float, str]],
                    start: float, end: float) -> list[str]:
    """Collect the phones belonging to one word interval.

    Membership is decided by the phone's midpoint rather than its boundaries: word and
    phone tiers are written independently and a shared boundary can differ in the last
    decimal.

    Args:
        phones: `(start, end, label)` of every phone interval of the utterance.
        start: Word interval start, seconds.
        end: Word interval end, seconds.

    Returns:
        Phone labels in time order, NFC-normalized with the length mark collapsed to match
        `common/phones_mfa.txt`. An empty label (silence) becomes ``"<SIL>"``.
    """
    out = []
    for phone_start, phone_end, label in phones:
        if start <= (phone_start + phone_end) / 2 < end:
            label = unicodedata.normalize('NFC', label) if label else SIL
            out.append(label.replace(LENGTH_MARK, ''))
    return out


def read_kept_ids(path: str) -> set[str] | None:
    """Utterance ids that survived the lab 1 filter, or None if the file is absent.

    The alignment was built on **raw** text, so it contains utterances a normalizer would
    reject — including three all-caps abbreviations whose transcriptions `mfa g2p` invented
    (`кпсс` → `k r ə f ɨ s̪`). Restricting to the lab 1 output removes them.
    """
    if not os.path.isfile(path):
        print(f'{path} not found: keeping every aligned utterance')
        return None
    with open(path, encoding='utf-8') as fh:
        return {line.split('|', 1)[0] for line in fh if line.strip()}


def read_alignment(align_root: str, keep: set[str] | None, keep_sil=False) -> list[dict]:
    """Read every TextGrid in `align_root` into per-token records.

    Args:
        align_root: Directory with `{id}.TextGrid` files.
        keep: Utterance ids to keep, or None for all.
        keep_sil: retains `<SIL>` tokens if true, else drops them

    Returns:
        One dict per aligned token with keys `id`, `token_id`, `token`, `phones`, `set`.
    """
    rows = []
    n_spn = 0  # считаем реплики, удаленные из-за spn
    for path in tqdm.tqdm(sorted(glob.glob(os.path.join(align_root, '*.TextGrid')))):
        utterance_id = os.path.basename(path)[:-len('.TextGrid')]
        if keep is not None and utterance_id not in keep:
            continue
        try:
            tg = textgrid.openTextgrid(path, True)
        except Exception as exc:
            print(f'Failed to read {utterance_id}: {exc}')
            continue
        word_tier, phone_tier = tg.tiers
        phones = [(p.start, p.end, p.label) for p in phone_tier.entries]

        # удаляем реплику целиком, если есть spn
        if any(label == 'spn' for _, _, label in phones):
            n_spn += 1
            continue
        
        fold = 'test' if int(utterance_id.split('_')[0]) % TEST_EVERY == 0 else 'train'

        token_id = 0
        for entry in word_tier.entries:
            if entry.label == '':
                if not keep_sil:
                    continue
                current_phones = '<SIL>'
                current_token = SIL  # токен <SIL> вместо пустой строки
            else:
                current_phones = ' '.join(phones_of_token(phones, entry.start, entry.end))
                current_token = unicodedata.normalize('NFC', entry.label)
            rows.append({
                'id': utterance_id,
                'token_id': token_id,
                'token': current_token,
                'phones': current_phones,
                'set': fold,
            })
            token_id += 1

    # статистика по удаленным spn
    if not keep_sil:
        print(f'Removed {n_spn} utterances with spn')
    
    return rows


def build_lexicon(rows: list[dict]) -> dict[str, Counter]:
    """Count pronunciations per token over the training fold only.

    A `spn` phone marks a word the aligner had no pronunciation for: the interval has a
    duration but no real transcription, so the row carries no lexical information.

    Building this over every fold instead would put the test words in the dictionary and
    make the reported PER meaningless.
    """
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row['set'] != 'train':
            continue
        phones = row['phones'].split()
        if not phones or 'spn' in phones:
            continue
        lexicon[row['token']][row['phones']] += 1
    return lexicon


def main() -> None:
    """Read the alignment, write the training table and the lexicon."""
    rows = read_alignment(ALIGN_DIR, read_kept_ids(NORMALIZED_META), keep_sil=False)
    utterances = {r['id'] for r in rows}
    test = {r['id'] for r in rows if r['set'] == 'test'}
    print(f'{len(rows)} tokens in {len(utterances)} utterances '
          f'({len(test)} test, {len(test) / max(len(utterances), 1):.1%})')

    os.makedirs(os.path.dirname(TOKENS_PATH), exist_ok=True)

    pd.DataFrame(rows).to_csv(TOKENS_PATH, encoding='utf-8', quoting=csv.QUOTE_NONE, sep='|', index=False, header=True)
    pd.DataFrame(
        read_alignment(ALIGN_DIR, read_kept_ids(NORMALIZED_META), keep_sil=True)
        ).to_csv(TOKENS_SIL_PATH, encoding='utf-8', quoting=csv.QUOTE_NONE, sep='|', index=False, header=True)
    
    
    lexicon = build_lexicon(rows)

    variants = sum(1 for counts in lexicon.values() if len(counts) > 1)
    print(f'lexicon from the train fold: {len(lexicon)} types, '
          f'{variants} with several pronunciations')

    with open(LEXICON_PATH, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, delimiter='|', quoting=csv.QUOTE_NONE)

        # изменены заголовки: было ['word', 'phones', 'count']
        writer.writerow(['token', 'phones', 'frequency'])
        for word in sorted(lexicon):
            for phones, count in lexicon[word].most_common():
                writer.writerow([word, phones, count])


if __name__ == '__main__':
    main()