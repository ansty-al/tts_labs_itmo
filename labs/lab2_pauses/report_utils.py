import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from cycler import cycler

BLUE, ORANGE, GREEN, GRAY, RED = '#4C78A8', '#F58518', '#54A24B', '#9AA5B1', '#E45756'

# русские названия классов, которые возвращает get_punctuation
PUNCT_RU = {'none': 'нет знака', 'comma': 'запятая', 'colon': 'двоеточие', 'semicolon': 'точка с запятой',
            'dash': 'тире', 'ellipsis': 'многоточие', 'exclamation': '!', 'period': 'точка',
            'question': '?', 'final': 'конец реплики'}

# порядок показа знаков на графике: от коротких пауз к длинным
PUNCT_ORDER = ['нет знака', 'запятая', 'двоеточие', 'тире', 'многоточие', '!', 'точка', '?']


def set_style():
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.bbox': 'tight',
        'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#555555',
        'axes.grid': True, 'axes.axisbelow': True, 'grid.color': '#DDDDDD', 'grid.linewidth': 0.8,
        'axes.prop_cycle': cycler(color=[BLUE, ORANGE, GREEN, RED, GRAY]),
        'legend.frameon': False,
    })


def plot_pause_hist(durations):

    ms = np.asarray(durations) * 1000
    short, mid, long_ = (ms < 100).mean() * 100, ((ms >= 100) & (ms < 400)).mean() * 100, (ms >= 400).mean() * 100

    counts, edges = np.histogram(ms[ms <= 1000], bins=np.arange(0, 1020, 20))
    centers = (edges[:-1] + edges[1:]) / 2
    colors = np.where(centers < 100, GRAY, np.where(centers < 400, BLUE, ORANGE))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(centers, counts, width=18, color=colors)
    ax.legend(handles=[Patch(color=GRAY, label=f'короткие (< 100 мс): {short:.0f}%'),
                       Patch(color=BLUE, label=f'средние (100–400 мс): {mid:.0f}%'),
                       Patch(color=ORANGE, label=f'длинные (≥ 400 мс): {long_:.0f}%')])
    ax.set_xlabel('Длительность паузы, мс')
    ax.set_ylabel('Количество пауз')
    ax.set_title('Паузы между словами (до 1 с)')
    plt.show()


def plot_unit_durations(words, letters, vowels):

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    d = words['duration'][words['duration'] <= 2]
    ax[0].hist(d, bins=60, color=BLUE)
    ax[0].axvline(words['duration'].mean(), color='#333333', linestyle='--', linewidth=1.2)
    ax[0].set_title('Слова')
    ax[0].set_xlabel('Длительность, с')
    ax[0].set_ylabel('Количество')

    d = letters['estimated_letter_duration']
    ax[1].hist(d[d <= 0.25], bins=60, color=GREEN)
    ax[1].axvline(d.mean(), color='#333333', linestyle='--', linewidth=1.2)
    ax[1].set_title('Буквы')
    ax[1].set_xlabel('Длительность, с')
    ax[1].set_ylabel('Количество')

    means = vowels.groupby('label')['duration'].mean().sort_values(ascending=False) * 1000
    bars = ax[2].bar(means.index, means.values, color=ORANGE)
    ax[2].bar_label(bars, fmt='%.0f', fontsize=9, padding=2)
    ax[2].set_title('Гласные: средняя длительность')
    ax[2].set_ylabel('мс')
    ax[2].grid(axis='x', visible=False)

    plt.tight_layout()
    plt.show()


def plot_punctuation(pauses):

    inner = pauses[~pauses['is_final_pause']]
    sizes = inner.groupby('punctuation').size()
    order = [p for p in PUNCT_ORDER if sizes.get(p, 0) >= 30]
    share = sizes / sizes.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars = ax.bar(order, [sizes[p] for p in order], color=BLUE)
    ax.bar_label(
        bars,
        labels=[f'{sizes[p]}\n({share[p]:.0f}%)' for p in order],
        padding=2,
        fontsize=9
    )

    ax.set_ylim(0, sizes[order].max() * 1.18)
    ax.set_ylabel('Количество пауз')
    ax.set_title('Паузы между словами по знакам препинания')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='x', visible=False)

    plt.tight_layout()
    plt.show()
