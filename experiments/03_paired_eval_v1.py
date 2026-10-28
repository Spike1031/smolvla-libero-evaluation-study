"""LeRobot 0.6.1：同一任务的配对评估；每次完成后独立提交结果。"""
import os
os.environ.update(MPLBACKEND="Agg", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def completed(root, sid, k, plan_hash):
    path = root / f"state_{sid:03d}" / f"k{k}" / "result.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (value.get("plan_hash") != plan_hash or value.get("state_id") != sid
            or value.get("n_action_steps") != k or value.get("status") != "completed"
            or type(value.get("success")) is not bool):
        raise RuntimeError(f"结果不符合当前计划，停止而不覆盖：{path}")
    return value


def summarize(root, plan):
    rows = []
    for sid in plan["state_ids"]:
        for k in plan["action_steps"]:
            r = completed(root, sid, k, digest(plan))
            if r is not None:
                rows.append(r)
    paired = [sid for sid in plan["state_ids"]
              if all(any(r["state_id"] == sid and r["n_action_steps"] == k for r in rows)
                     for k in plan["action_steps"])]
    groups = {}
    for k in plan["action_steps"]:
        all_rows = [r for r in rows if r["n_action_steps"] == k]
        pairs = [r for r in all_rows if r["state_id"] in paired]
        groups[str(k)] = {
            "completed": len(all_rows), "successes": sum(r["success"] for r in all_rows),
            "paired_n": len(pairs),
            "paired_success_rate": sum(r["success"] for r in pairs) / len(pairs) if pairs else None,
            "paired_eval_seconds": sum(r["eval_seconds"] for r in pairs),
        }
    summary = {"plan_hash": digest(plan), "total_completed": len(rows),
               "total_requested": len(plan["state_ids"]) * len(plan["action_steps"]),
               "paired_state_ids": paired, "groups": groups}
    write_json(root / "summary.json", summary)
    return summary


def run_pairs(root, plan, max_new, audit, **kw):
    import numpy as np
    import torch
    from lerobot.scripts import lerobot_eval as ev
    from lerobot.envs.libero import LiberoEnv
    tasks = [(g, t, env) for g, group in kw["envs"].items() for t, env in group.items()]
    assert len(tasks) == 1 and tasks[0][:2] == ("libero_spatial", 0)
    env = tasks[0][2]
    assert env.num_envs == 1 and hasattr(env, "envs"), "仅支持单个同步环境"
    base = env.envs[0].unwrapped
    assert isinstance(base, LiberoEnv) and base.init_states
    assert max(plan["state_ids"]) < len(base._init_states), "初始状态数量不足，禁止取模重复"
    states = np.asarray(base._init_states)
    state_hash = hashlib.sha256(states.tobytes()).hexdigest()
    compatibility = {"initial_states_sha256": state_hash, "shape": list(states.shape),
                     "dtype": str(states.dtype), "audit": audit}
    compat_path = root / "compatibility.json"
    if compat_path.exists():
        assert json.loads(compat_path.read_text()) == compatibility, "版本、源码、GPU或初始状态变化，停止续跑"
    else:
        write_json(compat_path, compatibility)
    policy = kw["policy"]
    count = 0
    for sid in plan["state_ids"]:
        # 交替每一对的运行顺序，避免总是某一设置先运行。
        order = plan["action_steps"] if sid % 2 == 0 else list(reversed(plan["action_steps"]))
        for k in order:
            if completed(root, sid, k, digest(plan)) is not None:
                continue
            if count >= max_new:
                return {"overall": summarize(root, plan)}
            folder = root / f"state_{sid:03d}" / f"k{k}"
            attempt = folder / ("attempt_" + uuid.uuid4().hex[:12])
            attempt.mkdir(parents=True)
            seed = plan["seed_base"] + sid
            write_json(attempt / "started.json", {"state_id": sid, "n_action_steps": k,
                "seed": seed, "plan_hash": digest(plan), "started_utc": datetime.now(timezone.utc).isoformat()})
            policy.config.n_action_steps = k
            # 官方 rollout 会调用 policy.reset()，按新的 k 重建动作队列。
            base.init_state_id = sid
            ev.set_seed(seed)
            for name in ["env_preprocessor", "env_postprocessor", "preprocessor", "postprocessor"]:
                processor = kw[name]
                if hasattr(processor, "reset"):
                    processor.reset()
            print(f"\n开始：初始状态 {sid}，执行步数 {k}，种子 {seed}", flush=True)
            steps = 0
            original_step = base.step
            def counted_step(action):
                nonlocal steps
                result = original_step(action)
                steps += 1
                return result
            base.step = counted_step
            try:
                torch.cuda.synchronize()
                info = ev.eval_policy(env=env, policy=policy,
                    **{name: kw[name] for name in ["env_preprocessor", "env_postprocessor", "preprocessor", "postprocessor"]},
                    n_episodes=1, max_episodes_rendered=1, videos_dir=attempt / "videos",
                    return_episode_data=False, start_seed=seed)
                torch.cuda.synchronize()
            finally:
                base.step = original_step
            assert base.init_state_id == sid + 1, "发生额外 reset，停止提交本次结果"
            assert len(info["per_episode"]) == 1
            ep = info["per_episode"][0]
            assert ep["seed"] == seed
            assert 0 < steps <= 280
            write_json(attempt / "eval_info.json", info)
            # 视频编码出错不应丢弃成功/失败数据，也不能因失败结果而重跑。
            videos = []
            for value in info.get("video_paths", []):
                video = Path(value)
                valid = video.exists() and video.stat().st_size > 0
                if valid:
                    check = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
                                           capture_output=True, text=True)
                    valid = check.returncode == 0 and not check.stderr.strip()
                videos.append({"path": str(video.relative_to(root)), "decode_ok": valid})
            result = {"status": "completed", "plan_hash": digest(plan), "state_id": sid,
                "n_action_steps": k, "seed": seed, "success": bool(ep["success"]),
                "control_steps": steps, "eval_seconds": info["aggregated"]["eval_s"],
                "videos": videos, "attempt": str(attempt.relative_to(root)),
                "completed_utc": datetime.now(timezone.utc).isoformat()}
            write_json(folder / "result.json", result)
            count += 1
            summary = summarize(root, plan)
            print(f"已保存：成功={result['success']}，控制步数={steps}；累计 {summary['total_completed']}/40", flush=True)
    return {"overall": summarize(root, plan)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--max-new", type=int, default=2)
    p.add_argument("--status", action="store_true")
    a = p.parse_args()
    root = Path(a.root)
    if a.status:
        if not (root / "plan.json").exists():
            print("新实验尚未开始。")
        else:
            print(json.dumps(summarize(root, json.loads((root / "plan.json").read_text())), indent=2, ensure_ascii=False))
        return
    assert 1 <= a.max_new <= 40
    assert md.version("lerobot") == "0.6.1", "本脚本仅核对过 LeRobot 0.6.1"
    import torch
    assert torch.cuda.is_available(), "需要 GPU"
    from huggingface_hub import HfApi, snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    expected = {"protocol": "paired_v1", "state_ids": list(range(20)), "action_steps": [1, 5],
                "seed_base": 42, "suite": "libero_spatial", "task_id": 0,
                "model_repo": "HuggingFaceVLA/smolvla_libero", "vlm_repo": "HuggingFaceTB/SmolVLM2-500M-Instruct",
                "assets_repo": "lerobot/libero-assets", "use_amp": False,
                "note": "每次任务重设随机种子；旧两组仅作探索记录，不混入统计"}
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert all(plan.get(k) == v for k, v in expected.items()), "计划已经存在且不一致，停止覆盖"
    else:
        api = HfApi()
        plan = dict(expected, model_revision=api.model_info(expected["model_repo"]).sha,
                    vlm_revision=api.model_info(expected["vlm_repo"]).sha,
                    assets_revision=api.dataset_info(expected["assets_repo"]).sha)
        write_json(plan_path, plan)
    if summarize(root, plan)["total_completed"] == 40:
        print("全部40次已完成，无需重新评估。")
        return
    # 固定远程仓库版本，续跑时重复下载会优先使用缓存。
    model = snapshot_download(plan["model_repo"], revision=plan["model_revision"])
    vlm = snapshot_download(plan["vlm_repo"], revision=plan["vlm_revision"])
    import libero.libero as libero_core
    snapshot_download(plan["assets_repo"], repo_type="dataset", revision=plan["assets_revision"],
                      local_dir=str(Path(libero_core.__file__).parent / "assets"), max_workers=4)
    from lerobot.scripts import lerobot_eval as ev
    from lerobot.envs.libero import LiberoEnv
    packages = {d.metadata["Name"].lower(): d.version for d in md.distributions() if d.metadata["Name"]}
    audit = {"python": platform.python_version(), "packages": packages,
             "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
             "eval_source_sha256": sha_file(inspect.getfile(ev)),
             "env_source_sha256": sha_file(inspect.getfile(LiberoEnv)),
             "runner_sha256": sha_file(__file__)}
    session = root / "sessions" / uuid.uuid4().hex[:12]
    session.mkdir(parents=True)
    write_json(session / "environment.json", audit)
    freeze = subprocess.check_output(["uv", "pip", "freeze", "--python", sys.executable], text=True)
    (session / "requirements.txt").write_text(freeze)
    if not (root / "requirements.txt").exists():
        (root / "requirements.txt").write_text(freeze)
    # 保留官方模型加载和处理流程，仅替换批量调度，单次任务仍用官方 eval_policy。
    ev.eval_policy_all = lambda **kw: run_pairs(root, plan, a.max_new, audit, **kw)
    sys.argv = [sys.argv[0], f"--policy.path={model}", f"--policy.vlm_model_name={vlm}",
        "--policy.device=cuda", "--policy.use_amp=false", "--policy.load_vlm_weights=false",
        "--policy.n_action_steps=1", "--env.type=libero", "--env.task=libero_spatial",
        "--env.task_ids=[0]", "--env.observation_height=256", "--env.observation_width=256",
        "--env.control_mode=relative", "--env.init_states=true", "--env.max_parallel_tasks=1",
        "--eval.batch_size=1", "--eval.n_episodes=1", "--eval.use_async_envs=false",
        "--seed=42", f"--output_dir={session}"]
    write_json(session / "command.json", sys.argv)
    ev.main()


if __name__ == "__main__":
    main()
