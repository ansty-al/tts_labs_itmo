import csv
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from textgrid import TextGrid
from tqdm.auto import tqdm

pd.set_option("display.max_rows", 100)

DATA = Path("data")
ALIGN = Path("../../data/RUSLAN_22050_align_v2")
RAW = Path("../../data/RUSLAN/metadata_RUSLAN_22200.csv")
META = Path("../../data/metadata_RUSLAN_22200_normalized.csv")
PHONES = DATA / "phones_mfa.txt"

LIMIT = 500  # мс: интервалы длиннее считаем ошибкой выравнивания

IPA = """
b bʲ  d dʲ  dʐ dʑ  dz dzʲ  f fʲ  ɡ ɡʲ  ɣ ɣʲ  j  k kʲ  l lʲ  m mʲ  n nʲ
p pʲ  r rʲ  s sʲ  ʂ ɕː  t tʲ  ts tsʲ  tʂ tɕ  v vʲ  x xʲ  z zʲ  ʐ ʑː
a æ  ɛ e  ɨ i  o ɵ  u ʉ  ə ɐ ɪ ʊ
"""

VOWELS = set("a e i o u ɐ ə ɛ ɨ ɪ ɵ ʉ ʊ æ".split())
REDUCED = set("ɐ ə ɪ ʊ".split())
VOWEL_LETTERS = re.compile(r"[аеёиоуыэюя]")
ABBR = re.compile(r"(?<![А-Яа-яЁё])(ул|прим|тов|г|кг)\.?", re.I)


# загрузка данных
def norm(s):
    return unicodedata.normalize("NFC", s)

    
def read_mfa_dict(path):
    d = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 1:
            word, *ph = line.split()
            phones = " ".join(ph)
        else:
            word, phones = parts[0], parts[-1]
        d.setdefault(word, set()).add(phones.strip())
    return d

    
def mfa_word_set(mfa_dict):
    return {w for w, prons in mfa_dict.items() if prons != {"spn"}}

    
def load_data(data=DATA):
    data = Path(data)
    tokens = pd.read_csv(data / "tokens_phones.csv", sep="|", keep_default_na=False)
    lex = pd.read_csv(data / "lexicon.csv", sep="|", keep_default_na=False)
    mfa_dict = read_mfa_dict(data / "russian_mfa.dict")
    return tokens, lex, mfa_dict

    
def overview(tokens, lex, mfa_dict):
    print(f"\nВ tokens_phones: {tokens['id'].nunique():,} текстов; {len(tokens):,} токенов")
    print("\n", tokens["set"].value_counts().to_dict())
    print(f"\nВ lexicon: {len(lex):,} слов, из которых {lex['token'].nunique():,} уникальные")
    print(f"\nВ russian_mfa.dict: {len(mfa_dict):,} слов\n")

    
def check_tokens_files(data=DATA):
    data = Path(data)
    t = pd.read_csv(data / "tokens_phones.csv", sep="|", quoting=csv.QUOTE_NONE, keep_default_na=False)
    s = pd.read_csv(data / "tokens_phones_sil.csv", sep="|", quoting=csv.QUOTE_NONE, keep_default_na=False)
    print(len(t), len(s), t.id.nunique())
    print("\nspn:", t.phones.str.contains("spn").sum(), "| пустых phones:", (t.phones == "").sum())
    print("\n<SIL> в tokens_phones:", (t.token == "<SIL>").sum(),
          "\n<SIL> в tokens_phones_sil:", (s.token == "<SIL>").sum())
    print("\nуникальных фонем:", len({p for ph in t.phones for p in ph.split()}))


    
# фонемный инвентарь, частоты
def phone_inventory(tokens):
    phone_counts = tokens["phones"].str.split().explode().value_counts()
    return pd.DataFrame({
        "count": phone_counts,
        "share_%": (phone_counts / phone_counts.sum() * 100).round(3),
    })

    
def plot_phone_counts(inventory):
    print(f"Всего фонем в корпусе: {len(inventory)}")
    fig, ax = plt.subplots(figsize=(12, 8))
    inventory["count"].plot.bar(ax=ax, color="steelblue", rot=0)
    ax.set_xlabel("Фонемы")
    ax.set_ylabel("Количество")
    ax.set_title("Частота фонем в RUSLAN")
    plt.show()

    
def strip_notation(ph):
    return unicodedata.normalize("NFC", ph).replace("\u032a", "")  # удаление дентального знака

    
def compare_with_ipa(inventory):
    ipa = set(IPA.split())
    corpus_ph = {strip_notation(p) for p in inventory.index}
    print(f"В корпусе: {len(corpus_ph)}, на странице IPA: {len(ipa)}, совпадает: {len(corpus_ph & ipa)}")
    print("\nЕсть в корпусе, но нет в IPA:", sorted(corpus_ph - ipa))
    print("\nЕсть в IPA, но нет в корпусе:", sorted(ipa - corpus_ph))



# длительности
def read_durations(align=ALIGN, cache=None):
    if cache and Path(cache).exists():
        return pd.read_pickle(cache)
    rows = []
    for f in tqdm(sorted(Path(align).glob("*.TextGrid"))):
        tg = TextGrid.fromFile(str(f))
        phones = next(t for t in tg.tiers if t.name == "phones")
        rows += [(f.stem, p.minTime, p.mark, (p.maxTime - p.minTime) * 1000)
                 for p in phones if p.mark and p.mark != "spn"]
    d = pd.DataFrame(rows, columns=["id", "start", "phone", "dur_ms"])
    if cache:
        d.to_pickle(cache)
    return d

    
def duration_stats(df):
    return (df.groupby("phone")["dur_ms"]
              .agg(n="count", mean="mean", std="std", median="median")
              .assign(cv=lambda x: x["std"] / x["mean"])
              .round(2)
              .sort_values("mean"))


def duration_tables(d, limit=LIMIT):
    bad = d["dur_ms"] > limit
    print(f"Интервалов > {limit} мс: {bad.sum()} из {len(d)} ({100 * bad.mean():.3f}%), "
          f"предложений: {d.loc[bad, 'id'].nunique()}")
    return duration_stats(d), duration_stats(d[~bad])


def outlier_ids(d, limit=LIMIT):
    bad_ids = sorted(d.loc[d["dur_ms"] > limit, "id"].unique())
    print(len(bad_ids), bad_ids[:5])
    return bad_ids


def outlier_examples(d, limit=LIMIT, n=5, meta=META, seed=0):
    bad = d[d["dur_ms"] > limit]
    print("Фонемы, чаще всего дающие выбросы:")
    print(bad["phone"].value_counts().head(10).to_string())
    text = (pd.read_csv(meta, sep="|", header=None, quoting=csv.QUOTE_NONE, dtype=str,
                        keep_default_na=False).set_index(0)[2])
    ids = bad["id"].drop_duplicates()
    for i in ids.sample(min(n, len(ids))):
        print(f"\n{i}: {text.get(i, '(нет в метаданных)')}")
        for _, r in bad[bad["id"] == i].iterrows():
            print(f"   фонема «{r['phone']}» с {r['start']:.2f} с, длительность {r['dur_ms']:.0f} мс")


def outlier_effect(stats, stats_clean, phones=None, top=5):
    cols = ["mean", "std", "cv"]
    if phones is None:
        change = (stats["std"] - stats_clean["std"]).abs().dropna()
        phones = change.nlargest(top).index
    phones = list(phones)
    return pd.concat({"с выбросами": stats.loc[phones, cols],
                      "без выбросов": stats_clean.loc[phones, cols]}, axis=1)


def plot_mean_durations(d, stats_clean, limit=LIMIT):
    s = stats_clean
    mean_all = d.loc[d["dur_ms"] <= limit, "dur_ms"].mean()

    fig, ax = plt.subplots(figsize=(8, 18))
    ax.barh(s.index, s["mean"], color="steelblue")
    ax.axvline(mean_all, color="gray", ls="--", label=f"среднее по корпусу ({mean_all:.0f} мс)")
    ax.set_xlabel("Средняя длительность, мс")
    ax.set_title("Средняя длительность фонем (без выбросов)")
    ax.tick_params(axis="y", labelsize=9)
    ax.margins(y=0.01)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    print(f"Средняя длительность по корпусу: {mean_all:.1f} мс")
    print("\nСамые вариативные (n >= 100):\n",
          s[s["n"] >= 100]["cv"].nlargest(5).to_string())


# редкие фонемы
def rare_phones(inventory, lex, n=10):
    ph_list = lex["phones"].str.split()

    def examples(ph, k=5):
        mask = ph_list.map(lambda l: ph in l)
        return ", ".join(lex[mask].nlargest(k, "frequency")["token"])

    rare = inventory.sort_values("count").head(n)[["count", "share_%"]].copy()
    rare["examples (train lexicon)"] = [examples(p) for p in rare.index]
    return rare



# OOV
def oov_table(tokens, lex, mfa_dict):
    mfa_words = mfa_word_set(mfa_dict)
    lex_words = set(lex["token"])

    def oov_row(df, split, label, vocab):
        toks = df["token"]
        mask = ~toks.isin(vocab)
        return {"часть": split, "словарь": label,
                "OOV токенов": mask.sum(), "% токенов": round(100 * mask.mean(), 2),
                "OOV типов": toks[mask].nunique(),
                "% типов": round(100 * toks[mask].nunique() / toks.nunique(), 2)}

    parts = {"весь корпус": tokens,
             "train": tokens[tokens["set"] == "train"],
             "test": tokens[tokens["set"] == "test"]}
    dicts = {"russian_mfa.dict": mfa_words,
             "lexicon.csv": lex_words,
             "ни в одном": mfa_words | lex_words}
    return pd.DataFrame([oov_row(df, split, label, vocab)
                         for split, df in parts.items()
                         for label, vocab in dicts.items()])


def oov_details(tokens, mfa_dict):
    mfa_words = mfa_word_set(mfa_dict)
    vocab_types = tokens["token"].drop_duplicates()
    oov_types = vocab_types[~vocab_types.isin(mfa_words)]

    freq = tokens["token"].value_counts()
    print(f"Типов в корпусе: {len(vocab_types)}, нет в MFA: {len(oov_types)} "
          f"({100 * len(oov_types) / len(vocab_types):.1f}%)")
    print(f"Токенов с такими словами: {freq[oov_types].sum()} из {len(tokens)} "
          f"({100 * freq[oov_types].sum() / len(tokens):.2f}%)")
    print(f"Из них встречаются 1 раз: {(freq[oov_types] == 1).sum()}")
    print(oov_types.sample(min(30, len(oov_types)), random_state=0).tolist())


# данные, удаленные фильтром
def _filter_sets(tokens):
    raw = pd.read_csv(RAW, sep="|", names=["id", "raw"], quoting=csv.QUOTE_NONE, dtype=str)
    kept = set(pd.read_csv(META, sep="|", header=None, quoting=csv.QUOTE_NONE, dtype=str)[0])
    final = set(tokens["id"])
    dropped = sorted(set(raw["id"]) - kept)
    return raw, kept, final, dropped


def filter_funnel(tokens):
    raw, kept, final, dropped = _filter_sets(tokens)
    print(f"В исходном корпусе: {len(raw)}")
    print(f"Удалено фильтром лр 1: {len(dropped)}")
    print(f"Удалено из-за spn (этап 1 данной лр): {len(kept - final)}")
    print(f"Осталось в tokens_phones.csv: {len(final)}")


def missing_textgrids(tokens, align=ALIGN):
    _, kept, _, _ = _filter_sets(tokens)
    have = {f.stem for f in Path(align).glob("*.TextGrid")}
    missing = sorted(kept - have)
    print(f"Файлов TextGrid: {len(have)}; прошли фильтр лабы 1, но без TextGrid: {missing}")
    return missing


def word_kind(w, abbr_stems=()):
    w = w.strip()
    if w in abbr_stems:
        return "сокращение"
    if re.search(r"\d", w):
        return "цифры"
    if len(w) > 2 and not VOWEL_LETTERS.search(w):
        return "нет гласных (аббревиатура)"
    return "обычное"


def find_suspicious_words(tokens, align=ALIGN):
    raw, _, _, dropped = _filter_sets(tokens)
    raw_text = raw.set_index("id")["raw"]
    rows = []
    for i in tqdm(dropped):
        path = Path(align) / f"{i}.TextGrid"
        if not path.exists():
            continue
        t = {x.name: x for x in TextGrid.fromFile(str(path)).tiers}
        stems = {s.lower() for s in ABBR.findall(raw_text[i])}
        for w in t["words"]:
            if not w.mark:
                continue
            k = word_kind(w.mark, stems)
            if k == "обычное":
                continue
            ph = [p.mark for p in t["phones"]
                  if p.mark and w.minTime <= (p.minTime + p.maxTime) / 2 < w.maxTime]
            rows.append((i, w.mark, k, " ".join(ph)))
    susp = pd.DataFrame(rows, columns=["id", "word", "kind", "phones"])
    susp["spn"] = susp["phones"].eq("spn")
    return susp


def suspicious_summary(susp):
    print(pd.crosstab(susp["kind"], susp["spn"].map({True: "spn", False: "есть транскрипция"})))


def fake_transcriptions(susp, n=10):
    return susp[~susp["spn"]].groupby(["kind"]).head(n)



# омографы и ударение
def vowels_only(phones):
    return " ".join(p for p in phones.split() if p in VOWELS)


def stress_pos(phones):
    """
    Номер слога с полным (нередуцированным) гласным - предполагаемое ударение.
    """
    v = [p for p in phones.split() if p in VOWELS]
    full = [i for i, p in enumerate(v) if p not in REDUCED]
    return full[0] if len(v) >= 2 and len(full) == 1 else None


def homograph_candidates(lex, min_minor=5, min_share=0.1):
    """
    Слова, у которых в lexicon.csv встречаются разные позиции ударения.

    `minor` - число употреблений нечастотных вариантов (заведомо ошибочно при «выбрать самый частотный»)
    устойчивые - те, у кого редкий вариант встречается не меньше `min_minor` раз и составляет не меньше `min_share` от всех употреблений.
    """
    st = lex.assign(stress=lex["phones"].map(stress_pos)).dropna(subset=["stress"])
    g = st.groupby(["token", "stress"])["frequency"].sum().groupby("token")
    cand = pd.DataFrame({"variants": g.size(), "total": g.sum(), "minor": g.sum() - g.max()})
    cand["minor_share"] = (cand["minor"] / cand["total"]).round(3)

    robust = (cand[(cand["variants"] > 1) & (cand["minor"] >= min_minor) & (cand["minor_share"] >= min_share)]
              .sort_values("minor", ascending=False))
    print(f"Многосложных слов с определимым ударением: {len(cand)}")
    print(f"С разными позициями ударения: {(cand['variants'] > 1).sum()}")
    print(f"Кандидатов в омографы (после порогов): {len(robust)}")
    all_h = cand[cand["variants"] > 1].sort_values("minor", ascending=False)
    return all_h, robust


def lex_variants(lex, word):
    return lex[lex["token"] == word].sort_values("frequency", ascending=False)[["phones", "frequency"]]


def show_dictionary_variants(words, lex, mfa_dict):
    for w in words:
        print(f"\n=== {w} ===")
        print("lexicon.csv:\n", lex_variants(lex, w).to_string(index=False))
        print("russian_mfa.dict:", sorted(mfa_dict.get(w, [])) or "нет в словаре")


def mfa_variants_report(homographs, mfa_dict):
    mfa_words = mfa_word_set(mfa_dict)

    def mfa_stress_variants(word):
        return {stress_pos(p) for p in mfa_dict.get(word, []) if stress_pos(p) is not None}

    h = homographs.assign(
        mfa_variants=[len(mfa_stress_variants(w)) for w in homographs.index],
        in_mfa=homographs.index.isin(mfa_words),
    )
    print(f"Кандидатов: {len(h)}")
    print(f"  нет в словаре MFA:                {(~h['in_mfa']).sum()}")
    print(f"  в словаре MFA >1 позиции ударения: {(h['mfa_variants'] > 1).sum()}")
    print(f"  в словаре MFA 1 позиция:           {(h['mfa_variants'] == 1).sum()}")


def show_contexts(tokens, words, n=3):
    """Предложения, в которых у слова есть оба варианта ударения"""
    sentences = tokens.sort_values(["id", "token_id"]).groupby("id")["token"].apply(" ".join)
    for w in words:
        print(f"\n======== {w} ========")
        t = tokens[tokens["token"] == w].copy()
        t["stress"] = t["phones"].map(stress_pos)
        for s, grp in t.dropna(subset=["stress"]).groupby("stress"):
            print(f"\n[ударение на слоге {int(s) + 1}] {grp['phones'].iloc[0]}  ({len(grp)} раз)")
            for _, r in grp.head(n).iterrows():
                print("  ", sentences[r["id"]])


def vowel_variants(lex, min_minor=5, min_share=0.1):
    """Слова, варианты которых в lexicon.csv различаются гласными (не только ударением)"""
    lx = lex.assign(vowels=lex["phones"].map(vowels_only))
    g = lx.groupby(["token", "vowels"])["frequency"].sum().groupby("token")
    cand = pd.DataFrame({"variants": g.size(), "total": g.sum(), "minor": g.sum() - g.max()})
    cand["minor_share"] = (cand["minor"] / cand["total"]).round(3)
    out = (cand[(cand["variants"] > 1) & (cand["minor"] >= min_minor) & (cand["minor_share"] >= min_share)]
           .sort_values("minor", ascending=False))
    n_train = lex["frequency"].sum()
    print(f"Слов, у которых гласные произносятся по-разному (редкий вариант не реже 5 раз и не реже 10% употреблений): {len(out)}; Токенов train, где стоит не самый частотный вариант (на них метод «самый частотный вариант» ошибется): {out['minor'].sum()} "
          f"из {n_train} ({100 * out['minor'].sum() / n_train:.2f}% токенов train)")
    return out



# целевой список фонем
def compare_with_target(d, inventory, phones=PHONES):
    target = {norm(p) for p in Path(phones).read_text(encoding="utf-8").split()}
    target.discard("<SIL>")

    raw_ph = {norm(p) for p in d["phone"].unique()}
    corpus_ph = {norm(p) for p in inventory.index}

    print(f"Целевой список: {len(target)} | сырые TextGrid: {len(raw_ph)} | tokens_phones.csv: {len(corpus_ph)}")
    print("\nСырые TextGrid, но нет в целевом:", sorted(raw_ph - target))
    print("tokens_phones.csv, но нет в целевом:", sorted(corpus_ph - target) or "нет")
    print("Целевой, но нет в tokens_phones.csv:", sorted(target - corpus_ph) or "нет")


def long_consonants_share(d):
    mask = d["phone"].str.contains("ː")
    print(f"Интервалов с долгой фонемой: {mask.sum()} из {len(d)} ({100 * mask.mean():.2f}%)")