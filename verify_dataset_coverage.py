# verify_dataset_coverage.py
"""
在晶創25（或任何機器）上跑這支，用來確認：
  1. 每個 split（train/valid/test）底下，noisy/clean 波形是否都能正確配對
     （用跟 spectrogram_dataset.py 完全相同的 collect_waveform_pairs 邏輯）
  2. 每個 split 實際涵蓋哪些受試者（subject id），筆數多少
  3. 跟 prepare_data.py 原本設計的預期值比對：
        train  -> subject 11–40（30人）
        valid  -> subject 1–10（10人）
        test   -> subject 1–10（10人）

用法：
    python3 verify_dataset_coverage.py --data_root semg_data/processed

不會動到任何資料，也不會轉檔，純讀取 + 統計。
"""
import argparse
import os
import re
import sys
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))

from spectrogram_dataset import collect_waveform_pairs

# prepare_data.py 裡寫死的預期值（EMGdata.__init__ / prepare()）
EXPECTED = {
    'train': {'subjects': set(range(11, 41)), 'exercise': 1, 'channels': {2}},
    'valid': {'subjects': set(range(1, 11)), 'exercise': 3, 'channels': {2}},
    'test':  {'subjects': set(range(1, 11)), 'exercise': 2, 'channels': {9, 10, 11, 12}},
}

# 檔名格式（來自 prepare_data.py）：S{subject}_E{exercise}_A1_ch{channel}_{seg}.npy
FNAME_RE = re.compile(r"^S(\d+)_E(\d+)_A1_ch(\d+)_(\d+)$")


def parse_fields(npy_path):
    """從檔名解析 subject / exercise / channel / segment。解析不出來回傳 None。"""
    stem = os.path.splitext(os.path.basename(npy_path))[0]
    m = FNAME_RE.match(stem)
    if not m:
        return None
    subject, exercise, channel, seg = map(int, m.groups())
    return {'subject': subject, 'exercise': exercise, 'channel': channel, 'segment': seg}


def verify_split(data_root, split):
    print(f"\n{'=' * 60}")
    print(f"Split: {split}")
    print('=' * 60)

    split_dir = os.path.join(data_root, split)
    try:
        pairs = collect_waveform_pairs(split_dir, 'noisy', 'clean', ext='.npy')
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print(f"配對成功筆數 (noisy, clean pairs): {len(pairs)}")

    subjects = set()
    exercises = set()
    channels = set()
    unparsed = 0

    for noisy_path, _ in pairs:
        fields = parse_fields(noisy_path)
        if fields is None:
            unparsed += 1
            continue
        subjects.add(fields['subject'])
        exercises.add(fields['exercise'])
        channels.add(fields['channel'])

    if unparsed:
        print(f"[Warning] {unparsed} 個檔名不符合預期格式 S{{sub}}_E{{ex}}_A1_ch{{ch}}_{{seg}}.npy，"
              f"無法解析、已跳過統計（不影響 pairs 本身的配對正確性）")

    print(f"實際涵蓋 subject 數量: {len(subjects)}")
    print(f"實際涵蓋 subject id : {sorted(subjects)}")
    print(f"實際涵蓋 exercise   : {sorted(exercises)}")
    print(f"實際涵蓋 channel    : {sorted(channels)}")

    exp = EXPECTED[split]
    print("\n--- 跟 prepare_data.py 預期值比對 ---")

    missing_subjects = exp['subjects'] - subjects
    extra_subjects = subjects - exp['subjects']
    if not missing_subjects and not extra_subjects:
        print(f"✔ subject 覆蓋率完全符合預期（{len(exp['subjects'])} 人）")
    else:
        if missing_subjects:
            print(f"✘ 缺少 subject: {sorted(missing_subjects)}")
        if extra_subjects:
            print(f"✘ 出現預期外的 subject: {sorted(extra_subjects)}")

    if exercises == {exp['exercise']}:
        print(f"✔ exercise 符合預期（E{exp['exercise']}）")
    else:
        print(f"✘ exercise 不符預期，預期 E{exp['exercise']}，實際 {sorted(exercises)}")

    if channels == exp['channels']:
        print(f"✔ channel 符合預期（{sorted(exp['channels'])}）")
    else:
        print(f"✘ channel 不符預期，預期 {sorted(exp['channels'])}，實際 {sorted(channels)}")

    # 每個 subject 的樣本數分佈，看有沒有某人資料明顯偏少（可能漏檔）
    per_subject_count = Counter()
    for noisy_path, _ in pairs:
        fields = parse_fields(noisy_path)
        if fields is not None:
            per_subject_count[fields['subject']] += 1
    if per_subject_count:
        counts = list(per_subject_count.values())
        print(f"\n每位 subject 的樣本數：min={min(counts)}, max={max(counts)}, "
              f"平均={sum(counts) / len(counts):.1f}")
        low = {s: c for s, c in per_subject_count.items() if c < 0.5 * (sum(counts) / len(counts))}
        if low:
            print(f"[Warning] 這些 subject 樣本數明顯偏少（可能漏檔），建議手動檢查: {low}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='semg_data/processed',
                    help='prepare_data.py 的輸出目錄（config 裡的 sEMG_dataset_dir）')
    p.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    args = p.parse_args()

    for split in args.splits:
        verify_split(args.data_root, split)

    print(f"\n{'=' * 60}")
    print("檢查完成。全部顯示 ✔ 代表資料切分跟涵蓋範圍與原作者 prepare_data.py 一致。")
    print('=' * 60)


if __name__ == '__main__':
    main()
