"""
查找 RoboTwin LeRobot 数据集中包含指定 task 的所有 episode。

注意：tasks.jsonl 里的 "task" 字段是自然语言指令（caption），
不是 RoboTwin 的"任务大类名"（如 adjust_bottle）。任务大类通常体现在
meta/episodes.jsonl 的额外字段里，或体现在数据集子目录命名里。

用法示例：
    # 1) 先列出 meta 文件，看看有哪些元信息可用（用于探查任务大类字段）
    python scripts/find_robotwin_episodes.py --list-meta

    # 2) 按 caption 关键词模糊匹配（默认大小写不敏感子串）
    python scripts/find_robotwin_episodes.py --caption bottle --show-frames

    # 3) 按 episodes.jsonl 中的 task_name / task / category 等字段匹配
    python scripts/find_robotwin_episodes.py --episode-task adjust_bottle

    # 4) 按正则匹配 caption
    python scripts/find_robotwin_episodes.py --caption "adjust.*bottle" --regex

    # 只输出 episode 索引（逗号分隔），方便直接贴到 ROBOTWIN_EPISODE_INDICES
    python scripts/find_robotwin_episodes.py --caption bottle --indices-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查找 RoboTwin LeRobot 数据集中匹配 task 的 episode")
    parser.add_argument(
        "--root",
        default="/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0",
        help="RoboTwin LeRobot 数据集根目录（包含 meta/info.json）",
    )
    parser.add_argument(
        "--list-meta",
        action="store_true",
        help="仅列出 meta/ 下的文件并打印 episodes.jsonl 首条记录，用于探查任务大类字段",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="按 caption（自然语言指令）匹配。默认走 episodes.jsonl 的 tasks 列表（快），"
             "任意一条指令命中即认为该 episode 命中；可加 --scan-parquet 切换到逐 episode 扫 parquet 的精确模式",
    )
    parser.add_argument(
        "--scan-parquet",
        action="store_true",
        help="在 --caption 模式下，改为扫描每条 episode 的 parquet 的 task_index 列做精确判断（更慢）",
    )
    parser.add_argument(
        "--episode-task",
        default=None,
        help="按 episodes.jsonl 中的任务大类字段匹配（如 adjust_bottle），脚本会自动尝试常见字段名",
    )
    parser.add_argument(
        "--episode-task-field",
        default=None,
        help="显式指定 episodes.jsonl 中要匹配的字段名（如 task_name / task / category 等）",
    )
    parser.add_argument("--regex", action="store_true", help="将 --caption / --episode-task 作为正则表达式匹配")
    parser.add_argument("--indices-only", action="store_true", help="仅输出 episode 索引（逗号分隔）")
    parser.add_argument("--show-frames", action="store_true", help="显示每条 episode 的帧数")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="分析模式：在 --caption 命中后，按 --bucket 大小分桶统计命中分布，自动找出 task 所在连续区间",
    )
    parser.add_argument(
        "--bucket",
        type=int,
        default=550,
        help="--analyze 模式下的桶大小，默认 550（即假设 50 个 task × 550 demo/task = 27500）",
    )
    parser.add_argument(
        "--peek",
        type=int,
        default=None,
        help="打印若干 episode 的 caption 用于人工核对，例如 --peek 10 会打印前 10 个、桶边界附近、末尾的 caption",
    )
    return parser.parse_args()


def make_matcher(pattern: str, use_regex: bool):
    if use_regex:
        regex = re.compile(pattern)
        return lambda s: bool(regex.search(s))
    pat = pattern.lower()
    return lambda s: pat in s.lower()

def load_tasks(root: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    tasks_path = root / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        sys.exit(f"[ERROR] 未找到 tasks.jsonl: {tasks_path}")
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks[int(item["task_index"])] = str(item["task"])
    return tasks


def load_episodes_jsonl(root: Path) -> list[dict]:
    """读取 meta/episodes.jsonl（如存在），返回每行 dict 的列表，按 episode_index 升序对齐。"""
    path = root / "meta" / "episodes.jsonl"
    if not path.is_file():
        return []
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] 解析 episodes.jsonl 失败: {e}", file=sys.stderr)
    return items


def list_meta(root: Path) -> None:
    meta_dir = root / "meta"
    print(f"[INFO] 数据集根目录: {root}")
    print(f"[INFO] meta 目录: {meta_dir}")
    if not meta_dir.is_dir():
        sys.exit(f"[ERROR] 未找到 meta 目录: {meta_dir}")
    print(f"[INFO] meta 目录下文件:")
    for p in sorted(meta_dir.iterdir()):
        try:
            size = p.stat().st_size
        except Exception:
            size = -1
        print(f"    {p.name}    {size} bytes")

    # info.json
    info_path = meta_dir / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        print(f"\n[INFO] info.json 关键字段:")
        for k in ("total_episodes", "total_frames", "fps", "chunks_size", "data_path", "video_path"):
            if k in info:
                print(f"    {k}: {info[k]}")

    # episodes.jsonl 首条
    episodes = load_episodes_jsonl(root)
    if episodes:
        print(f"\n[INFO] episodes.jsonl 共 {len(episodes)} 条，首条记录字段:")
        first = episodes[0]
        for k, v in first.items():
            sv = str(v)
            if len(sv) > 200:
                sv = sv[:200] + "..."
            print(f"    {k}: {sv}")
        # 把所有出现过的 key 集合也打出来
        all_keys = set()
        for e in episodes[:5000]:
            all_keys.update(e.keys())
        print(f"\n[INFO] episodes.jsonl 前 5000 条出现过的字段名: {sorted(all_keys)}")
    else:
        print("\n[INFO] 未找到 meta/episodes.jsonl")

    # tasks.jsonl 前几条
    tasks = load_tasks(root)
    print(f"\n[INFO] tasks.jsonl 共 {len(tasks)} 条，前 5 条:")
    for i, (idx, name) in enumerate(sorted(tasks.items())):
        if i >= 5:
            break
        print(f"    {idx}: {name}")


def pick_episode_task_value(record: dict, field: str | None) -> str | None:
    """从 episodes.jsonl 单条记录中提取任务大类字符串。

    若用户指定了 field，则只看该 field；否则按常见字段名顺序尝试。
    """
    candidates = (
        [field]
        if field
        else [
            "task_name",
            "task",
            "task_category",
            "category",
            "task_class",
            "tasks",  # 有可能是列表
        ]
    )
    for key in candidates:
        if key is None or key not in record:
            continue
        v = record[key]
        if isinstance(v, str):
            return v
        if isinstance(v, (list, tuple)):
            return " ".join(str(x) for x in v)
    return None


def peek_episodes(root: Path, n: int, bucket: int, total_episodes: int) -> None:
    """打印若干代表性 episode 的 caption，用于人工核对 task 边界。

    打印：
      - 前 n 条（验证第 0 个 task 的样子）
      - 每个桶边界 [k*bucket-2, k*bucket+2] 的 caption（验证 task 切换点）
      - 末 n 条
    """
    episodes = load_episodes_jsonl(root)
    if not episodes:
        sys.exit("[ERROR] 未找到 meta/episodes.jsonl")
    by_idx = {int(e["episode_index"]): e for e in episodes}

    def show(ep_idx: int, prefix: str = "") -> None:
        e = by_idx.get(ep_idx)
        if e is None:
            return
        captions = e.get("tasks") or []
        first_cap = captions[0] if isinstance(captions, list) and captions else str(captions)
        cap = first_cap if len(first_cap) <= 90 else first_cap[:90] + "..."
        length = e.get("length", "?")
        print(f"  {prefix}ep={ep_idx:>5d}  len={length:>4}  cap[0]={cap!r}")

    print(f"[INFO] total_episodes={total_episodes}, bucket={bucket}")
    print(f"\n[INFO] 前 {n} 条 episode 的 caption[0]:")
    for i in range(min(n, total_episodes)):
        show(i)

    n_buckets = total_episodes // bucket
    print(f"\n[INFO] 桶边界附近的 caption[0]（共 {n_buckets} 个桶，桶大小 {bucket}）:")
    for k in range(1, n_buckets):
        boundary = k * bucket
        print(f"  ---- boundary at ep={boundary} (k={k}) ----")
        for off in range(-2, 3):
            ep = boundary + off
            if 0 <= ep < total_episodes:
                show(ep, prefix=f"[Δ{off:+d}] ")

    print(f"\n[INFO] 末 {n} 条 episode 的 caption[0]:")
    for i in range(max(0, total_episodes - n), total_episodes):
        show(i)


def analyze_distribution(indices: list[int], total_episodes: int, bucket: int) -> None:
    """分析命中 episode 的分桶分布，找连续区间，并校验是否对齐桶边界。"""
    print("\n" + "=" * 70)
    print("[ANALYZE] 命中 episode 的分布分析")
    print("=" * 70)
    n = len(indices)
    if n == 0:
        print("[ANALYZE] 命中数为 0，无可分析。")
        return
    print(f"  命中总数: {n}")
    print(f"  最小 ep:  {indices[0]}")
    print(f"  最大 ep:  {indices[-1]}")
    print(f"  跨度:    {indices[-1] - indices[0] + 1}")

    # 1) 找连续区间
    runs: list[tuple[int, int]] = []  # [start, end]
    s = indices[0]
    p = indices[0]
    for x in indices[1:]:
        if x == p + 1:
            p = x
        else:
            runs.append((s, p))
            s = p = x
    runs.append((s, p))
    print(f"\n  连续区间 ({len(runs)} 段):")
    for s, e in runs:
        print(f"    [{s}, {e}]    长度 = {e - s + 1}")

    # 2) 按 bucket 大小分桶统计
    n_buckets = (total_episodes + bucket - 1) // bucket
    bucket_hit = [0] * n_buckets
    for x in indices:
        bk = x // bucket
        if 0 <= bk < n_buckets:
            bucket_hit[bk] += 1
    print(f"\n  按桶大小 {bucket} 分桶（共 {n_buckets} 桶），各桶命中数:")
    nonzero = [(k, c) for k, c in enumerate(bucket_hit) if c > 0]
    for k, c in nonzero:
        bar = "#" * min(60, int(c / max(1, max(bucket_hit)) * 60))
        ratio = c / bucket * 100
        print(f"    bucket[{k:>2d}]  ep=[{k*bucket}, {(k+1)*bucket}) cnt={c:>4d} ({ratio:5.1f}%) {bar}")

    # 3) 路径 B 验证：是否存在某个桶 k 命中数 == bucket（即整段对齐）
    perfect = [k for k, c in enumerate(bucket_hit) if c == bucket]
    if perfect:
        print(f"\n  ✅ 完美对齐的桶 (命中数==bucket): {perfect}")
        for k in perfect:
            print(f"     -> task 范围: ep ∈ [{k*bucket}, {(k+1)*bucket}) (共 {bucket} 条)")
    else:
        print("\n  ⚠️ 没有任何桶完美对齐。可能原因：")
        print("     - bucket 大小猜错了（试试 --bucket 100 / 275 / 500）")
        print("     - 该 task 的 caption 关键词覆盖不全或有误匹配")
        print("     - 数据集打包顺序不是按 task 大类连续排列")
        # 给出最佳近似
        if nonzero:
            best_k, best_c = max(nonzero, key=lambda x: x[1])
            print(f"     最高命中桶: bucket[{best_k}]  cnt={best_c} ({best_c/bucket*100:.1f}%)")


def find_by_caption_via_episodes_jsonl(root: Path, pattern: str, use_regex: bool):
    """快路径：直接在 episodes.jsonl 的 tasks 列表里匹配自然语言指令。

    任一条指令命中即认为该 episode 命中。返回 (episode_index, -1, hit_caption, length)。
    """
    episodes = load_episodes_jsonl(root)
    if not episodes:
        sys.exit("[ERROR] 未找到 meta/episodes.jsonl，无法走快路径，请加 --scan-parquet")
    matcher = make_matcher(pattern, use_regex)

    print(f"[INFO] 在 episodes.jsonl 的 tasks 列表中匹配 caption: pattern={pattern!r} regex={use_regex}")
    matched: list[tuple[int, int, str, int]] = []
    for e in episodes:
        ep_idx = int(e.get("episode_index", -1))
        length = int(e.get("length", 0) or 0)
        captions = e.get("tasks") or []
        if not isinstance(captions, (list, tuple)):
            captions = [str(captions)]
        hit_caption: str | None = None
        for c in captions:
            if isinstance(c, str) and matcher(c):
                hit_caption = c
                break
        if hit_caption is None:
            continue
        matched.append((ep_idx, -1, hit_caption, length))
    matched.sort(key=lambda x: x[0])
    print(f"[INFO] 命中 {len(matched)} 条 episode")
    return matched


def find_by_caption(root: Path, info: dict, tasks: dict[int, str], pattern: str, use_regex: bool):
    """通过扫描每条 episode 的 parquet 中 task_index 列，匹配 tasks.jsonl 中含关键词的 caption。"""
    matcher = make_matcher(pattern, use_regex)
    matched_task_indices = {idx for idx, name in tasks.items() if matcher(name)}
    if not matched_task_indices:
        sys.exit(f"[ERROR] tasks.jsonl 中没有 caption 匹配 {pattern!r} (regex={use_regex})")

    chunks_size = int(info.get("chunks_size", 1000))
    total_episodes = int(info["total_episodes"])
    data_path_template = info["data_path"]

    print(f"[INFO] caption 匹配到 {len(matched_task_indices)} 个 task_index")
    print(f"[INFO] 开始扫描 {total_episodes} 条 episode 的 parquet ...")

    matched_episodes: list[tuple[int, int, str, int]] = []
    for ep_idx in range(total_episodes):
        ep_chunk = ep_idx // chunks_size
        rel = data_path_template.format(episode_chunk=ep_chunk, episode_index=ep_idx)
        data_path = root / rel
        if not data_path.is_file():
            continue
        try:
            table = pq.read_table(data_path, columns=["task_index"])
            tids = table["task_index"].to_pylist()
            if not tids:
                continue
            unique_tids = {int(t) for t in tids}
            hit = unique_tids & matched_task_indices
            if not hit:
                continue
            primary_tid = int(tids[0])
            matched_episodes.append(
                (ep_idx, primary_tid, tasks.get(primary_tid, "<unknown>"), len(tids))
            )
        except Exception as e:
            print(f"[WARN] 读取失败 {data_path}: {e}", file=sys.stderr)
        if (ep_idx + 1) % 500 == 0:
            print(
                f"    ...已扫描 {ep_idx + 1}/{total_episodes}, 当前命中 {len(matched_episodes)} 条",
                file=sys.stderr,
            )
    return matched_episodes


def find_by_episode_task(root: Path, pattern: str, use_regex: bool, field: str | None):
    """通过 meta/episodes.jsonl 的任务大类字段（如 task_name=adjust_bottle）匹配 episode。"""
    episodes = load_episodes_jsonl(root)
    if not episodes:
        sys.exit(
            f"[ERROR] 未找到 meta/episodes.jsonl。先用 --list-meta 查看可用元信息，"
            f"或改用 --caption 在 tasks.jsonl 中按指令文本匹配。"
        )
    matcher = make_matcher(pattern, use_regex)

    # 自动选 field：把 field 命中的统计输出，方便排查
    used_field: str | None = field
    matched: list[dict] = []
    if used_field is None:
        # 尝试每个候选字段，挑命中数最多的
        candidate_fields = ["task_name", "task", "task_category", "category", "task_class", "tasks"]
        best_field = None
        best_hits: list[dict] = []
        for k in candidate_fields:
            hits = [e for e in episodes if isinstance(e.get(k), (str, list, tuple)) and matcher(
                e[k] if isinstance(e[k], str) else " ".join(str(x) for x in e[k])
            )]
            if len(hits) > len(best_hits):
                best_field = k
                best_hits = hits
        used_field = best_field
        matched = best_hits
        if used_field is None or not matched:
            sample_keys = sorted({k for e in episodes[:200] for k in e.keys()})
            sys.exit(
                f"[ERROR] 在 episodes.jsonl 中未匹配到 {pattern!r}。可用字段示例: {sample_keys}。"
                f" 可使用 --episode-task-field 显式指定字段。"
            )
        print(f"[INFO] 自动选择字段 {used_field!r}, 命中 {len(matched)} 条 episode")
    else:
        for e in episodes:
            v = e.get(used_field)
            if isinstance(v, str) and matcher(v):
                matched.append(e)
            elif isinstance(v, (list, tuple)) and matcher(" ".join(str(x) for x in v)):
                matched.append(e)
        if not matched:
            sys.exit(f"[ERROR] episodes.jsonl 字段 {used_field!r} 中未匹配到 {pattern!r}")
        print(f"[INFO] 按指定字段 {used_field!r} 命中 {len(matched)} 条 episode")

    # 整理为 (episode_index, task_index, task_name, num_frames)
    out: list[tuple[int, int, str, int]] = []
    for e in matched:
        ep_idx = int(e.get("episode_index", -1))
        length = int(e.get("length", e.get("num_frames", 0)) or 0)
        # 取一个可读的 task 名：优先 used_field，其次 caption 列表
        v = e.get(used_field)
        name = v if isinstance(v, str) else (" | ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else "<unknown>")
        out.append((ep_idx, -1, name, length))
    out.sort(key=lambda x: x[0])
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"[ERROR] 数据集根目录不存在: {root}")

    if args.list_meta:
        list_meta(root)
        return

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        sys.exit(f"[ERROR] 未找到 info.json: {info_path}")
    info = json.loads(info_path.read_text())

    if not args.caption and not args.episode_task:
        sys.exit(
            "[ERROR] 请指定匹配方式之一：\n"
            "    --list-meta                           先查看 meta 文件结构\n"
            "    --caption bottle                       按 tasks.jsonl 指令文本子串匹配\n"
            "    --episode-task adjust_bottle           按 episodes.jsonl 任务大类字段匹配\n"
            "    --episode-task adjust_bottle --episode-task-field task_name\n"
        )

    if args.peek is not None:
        peek_episodes(root, args.peek, args.bucket, int(info["total_episodes"]))
        return

    if args.episode_task:
        matched_episodes = find_by_episode_task(
            root, args.episode_task, args.regex, args.episode_task_field
        )
    elif args.scan_parquet:
        tasks = load_tasks(root)
        matched_episodes = find_by_caption(root, info, tasks, args.caption, args.regex)
    else:
        matched_episodes = find_by_caption_via_episodes_jsonl(root, args.caption, args.regex)

    indices = sorted({m[0] for m in matched_episodes})

    if args.analyze:
        analyze_distribution(indices, int(info["total_episodes"]), args.bucket)

    if args.indices_only:
        print(",".join(str(i) for i in indices))
        return

    print(f"\n[RESULT] 共匹配到 {len(indices)} 条 episode")
    print(f"[RESULT] episode 索引列表（可直接用于 ROBOTWIN_EPISODE_INDICES）:")
    print(",".join(str(i) for i in indices))
    if args.show_frames:
        print("\n[RESULT] 详细列表 (episode_index, task_index, task_name, num_frames):")
        for ep_idx, tid, name, nf in matched_episodes:
            short = name if len(name) <= 80 else name[:80] + "..."
            print(f"    ep={ep_idx:>6d}  task_index={tid:>5d}  frames={nf:>5d}  task={short!r}")


if __name__ == "__main__":
    main()