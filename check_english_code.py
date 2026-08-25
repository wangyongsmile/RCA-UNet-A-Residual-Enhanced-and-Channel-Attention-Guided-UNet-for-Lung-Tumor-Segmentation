#!/usr/bin/env python3
"""Verify that public text/code files contain no CJK characters."""
from pathlib import Path
import re
import sys

CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]')
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.rst', '.yml', '.yaml', '.json', '.toml', '.ini', '.cfg', '.csv', '.sh', '.bat', '.ps1'}
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'results', 'trained_models', 'checkpoints'}

findings = []
root = Path(__file__).resolve().parent
for path in root.rglob('*'):
    if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
        continue
    if any(part in SKIP_DIRS for part in path.parts):
        continue
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        continue
    for line_no, line in enumerate(text.splitlines(), 1):
        if CJK.search(line):
            findings.append((path.relative_to(root), line_no, line.strip()))

if findings:
    print('FAIL: non-English CJK text was found:')
    for path, line_no, line in findings:
        print(f'{path}:{line_no}: {line}')
    sys.exit(1)

print('PASS: no CJK characters were found in public text/code files.')
