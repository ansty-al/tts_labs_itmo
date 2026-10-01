import csv
import re

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'

# знаки препинания, после которых может находиться пауза
PUNCT_CHARS = ',.!?:—–…'

# выделяем слово и пунктуацию в конце токена
TOKEN_RE = re.compile(r'^([^\w]*)(.*?\w)([^\w]*)$')

def split_token(token):
    
    """
    Разделяет токен на слово и идущую после него пунктуацию:

        'чувством,' -> ('чувством', ',')
        'перо.'     -> ('перо', '.')
        'словом'    -> ('словом', '')
    """
    
    token = str(token).strip()

    match = TOKEN_RE.match(token)

    if match is None:
        return '', token.replace(' ', '')

    word = match.group(2)
    punctuation = match.group(3).replace(' ', '')

    return word, punctuation

def main_punct(token):
    
    """
    Возвращает первый знак препинания в конце токена.
    Если пунктуации нет, возвращает пустую строку.
    """
    
    _, punctuation = split_token(token)

    for char in punctuation:
        if char in PUNCT_CHARS:
            return char

    return ''

class PausePredictor():

    def __init__(self):
        
        # медианная длительность паузы для каждого знака препинания
        self.median_duration = {}

        # для знаков, у которых в train не найдется статистики
        self.default_duration = 0.0

    def fit(self, df):
        
        """
        Обучение на train:

        - если после токена есть знак препинания, предполагаем паузу
        - длительность берем как медиану пауз после такого знака
        - последний токен не используется
        
        """

        train = df[
            (df['set'] == 'train') &
            (df['is_last_word'] == 0) &
            (df['is_pause_after'] == 1)
        ].copy()

        # определяем знак препинания для каждого токена
        train['punctuation'] = train['label_raw'].map(main_punct)

        # только паузы, связанные со знаком препинания
        train = train[train['punctuation'] != '']

        # медианная длительность паузы для каждого знака
        self.median_duration = (
            train
            .groupby('punctuation')['pause_duration']
            .median()
            .to_dict()
        )

        # для знака, которого не было в train
        if len(train):
            self.default_duration = float(
                train['pause_duration'].median()
            )

        return self

    def predict(self, tokens):
        
        """
        Предсказывает наличие и длительность паузы после каждого токена:

            is_pause       — массив 0/1
            pause_duration — массив длительностей в секундах
            
        """

        n = len(tokens)

        is_pause = np.zeros(n, dtype=int)
        pause_duration = np.zeros(n, dtype=float)

        if n == 0:
            return is_pause, pause_duration

        for i, token in enumerate(tokens):

            punctuation = main_punct(token)

            if punctuation == '':
                continue

            # после последнего токена паузу не предсказываем
            if i == n - 1:
                continue

            is_pause[i] = 1

            pause_duration[i] = self.median_duration.get(
                punctuation,
                self.default_duration
            )

        return is_pause, pause_duration

    def predict_durations(self, tokens):
        
        """
        Добавляем <SIL> после токенов, для которых предсказана пауза
        
        """
        
        is_pause, durations = self.predict(tokens)

        tokens_with_pauses = []
        durations_with_pauses = []

        for token, pause, duration in zip(
            tokens,
            is_pause,
            durations
        ):
            tokens_with_pauses.append(token)
            durations_with_pauses.append(-1.0)

            if pause:
                tokens_with_pauses.append('<SIL>')
                durations_with_pauses.append(duration)

        return (
            np.array(tokens_with_pauses),
            np.array(durations_with_pauses, dtype=np.float32)
        )


def load_data():

    return pd.read_csv(
        PAUSE_PREDICTOR_DATA,
        sep='|',
        quoting=csv.QUOTE_NONE,
        dtype={
            'id': str,
            'label': str,
            'label_raw': str
        },
        keep_default_na=False
    )


def predict_dataframe(predictor, pause_df):

    pause_df = pause_df.copy()

    is_pause_hat = np.zeros(len(pause_df), dtype=int)
    pause_duration_hat = np.zeros(len(pause_df), dtype=float)

    for positions in tqdm.tqdm(
        pause_df.groupby('id', sort=False).indices.values()
    ):
        tokens = pause_df.iloc[positions]['label_raw'].values

        predicted_pause, predicted_duration = predictor.predict(tokens)

        is_pause_hat[positions] = predicted_pause
        pause_duration_hat[positions] = predicted_duration

    pause_df['is_pause_hat'] = is_pause_hat
    pause_df['pause_duration_hat'] = pause_duration_hat

    return pause_df


def calc_metrics(df):

    precision = precision_score(
        df['is_pause_after'],
        df['is_pause_hat'],
        zero_division=0
    )

    recall = recall_score(
        df['is_pause_after'],
        df['is_pause_hat'],
        zero_division=0
    )

    f1 = f1_score(
        df['is_pause_after'],
        df['is_pause_hat'],
        zero_division=0
    )

    true_positive = df[
        (df['is_pause_after'] == 1) &
        (df['is_pause_hat'] == 1)
    ]

    if len(true_positive):
        mae = mean_absolute_error(
            true_positive['pause_duration'],
            true_positive['pause_duration_hat']
        )
    else:
        mae = float('nan')

    print(
        f'PRC: {precision:.3f}, '
        f'REC: {recall:.3f}, '
        f'F1: {f1:.3f}; '
        f'MAE: {mae:.3f}'
    )


def test_pause_predictor():

    pause_df = load_data()
    train_df = pause_df[pause_df['set'] == 'train']
    predictor = PausePredictor()

    # обучаем только на train
    predictor.fit(train_df)

    # предсказания для всего корпуса
    pause_df = predict_dataframe(predictor, pause_df)

    print('\nCalculate metrics, training fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df['set'] == 'train') & (pause_df['is_last_word'] == 0)])

    print('\nCalculate metrics, testing fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df['set'] == 'test') & (pause_df['is_last_word'] == 0)])

if __name__ == '__main__':
    test_pause_predictor()