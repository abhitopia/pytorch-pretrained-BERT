import pickle
from pathlib import Path


file = Path('./data/freshly/train.pkl')
trg_file = Path('./data/freshly/train.txt.raw')

data = pickle.load(file.open('rb'))

lines = []
for dialgue in data:
    for u in dialgue.utterances:
        lines.append(u.utterance)


with trg_file.open('tw', encoding='utf-8') as f:
    f.writelines(lines)


