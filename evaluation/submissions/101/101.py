import sys
import random
from pathlib import Path

_, test_questions_path, task1_pred_path, task2_pred_path = sys.argv

task1_preds, task2_preds = [], []
with open(test_questions_path, 'r', encoding='utf16') as f:
	questions = f.readlines()
	task1_preds = [str(random.randint(1, 1000)) for q in questions]
	task2_preds = [q.split()[0] for q in questions]

with open(Path(task1_pred_path) / '101.txt', 'w', encoding='utf16') as f:
	f.write('\n'.join(task1_preds))

with open(Path(task2_pred_path) / '101.txt', 'w', encoding='utf16') as f:
	f.write('\n'.join(task2_preds))