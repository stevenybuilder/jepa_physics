"""Data QA for all three datasets -> results/qa_{dataset}.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import DATASETS, qa  # noqa: E402

for dataset in DATASETS:
    r = qa(dataset)
    print(f"{dataset}: n={r['n_clips']} visible_frac mean={r['visible_frac']['mean']:.4f} "
          f"min={r['visible_frac']['min']:.4f} all_visible={r['visible_frac']['n_all_frames_visible']} "
          f"any_absent={r['n_clips_any_frame_disk_absent']} none_visible={r['visible_frac']['n_no_frame_visible']} "
          f"unique_hashes={r['n_unique_frame_hashes']} dup_groups={len(r['duplicate_groups'])} "
          f"dup_conflicting={len(r['duplicate_groups_with_conflicting_labels'])}")
