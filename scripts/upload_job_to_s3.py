#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resume-safe upload of a srtctl job log directory to S3.

Mirrors the upload path used by postprocess in do_sweep:
  s3://{bucket}/{prefix}/{YYYY-MM-DD}/{job_id}/

Before upload, generates benchmark-rollup.json using the benchmark-specific
rollup.py script (same as postprocess in do_sweep).

Uses `aws s3 sync` when available (skips unchanged files), with a boto3 fallback.

Examples:
  python scripts/upload_job_to_s3.py outputs/12345
  python scripts/upload_job_to_s3.py outputs/12345/logs --date 2026-05-26
  python scripts/upload_job_to_s3.py outputs/12345 --dry-run
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \\
    python scripts/upload_job_to_s3.py outputs/12345 --bucket my-bucket

Dependencies:
  pip install pyyaml
  awscli (recommended) OR boto3
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("Missing dependency: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


@dataclass(frozen=True)
class S3Settings:
    bucket: str
    prefix: str = "srtslurm"
    region: str | None = None
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None


def find_cluster_config(explicit: Path | None) -> Path | None:
    if explicit:
        return explicit if explicit.exists() else None

    env_config = os.environ.get("SRTSLURM_CONFIG")
    if env_config:
        path = Path(env_config)
        return path if path.exists() else None

    for candidate in (
        Path.cwd() / "srtslurm.yaml",
        Path.cwd().parent / "srtslurm.yaml",
        Path.cwd().parent.parent / "srtslurm.yaml",
        Path(__file__).resolve().parent.parent / "srtslurm.yaml",
    ):
        if candidate.exists():
            return candidate
    return None


def load_cluster_config(cluster_config_path: Path | None) -> dict[str, Any]:
    if not cluster_config_path:
        return {}

    with cluster_config_path.open() as handle:
        return yaml.safe_load(handle) or {}


def load_s3_settings(cluster_config: dict[str, Any]) -> S3Settings | None:
    s3_dict = (cluster_config.get("reporting") or {}).get("s3")
    if not s3_dict or not s3_dict.get("bucket"):
        return None

    return S3Settings(
        bucket=s3_dict["bucket"],
        prefix=s3_dict.get("prefix") or "srtslurm",
        region=s3_dict.get("region"),
        endpoint_url=s3_dict.get("endpoint_url"),
        access_key_id=os.path.expandvars(s3_dict["access_key_id"]) if s3_dict.get("access_key_id") else None,
        secret_access_key=os.path.expandvars(s3_dict["secret_access_key"]) if s3_dict.get("secret_access_key") else None,
    )


def find_benchmark_scripts_dir(cluster_config: dict[str, Any]) -> Path:
    srtctl_root = cluster_config.get("srtctl_root")
    if srtctl_root:
        scripts_dir = Path(os.path.expandvars(srtctl_root)) / "src/srtctl/benchmarks/scripts"
        if scripts_dir.is_dir():
            return scripts_dir

    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "src/srtctl/benchmarks/scripts"


def load_benchmark_type(job_dir: Path) -> str | None:
    config_path = job_dir / "config.yaml"
    if not config_path.exists():
        return None

    with config_path.open() as handle:
        raw = yaml.safe_load(handle) or {}

    benchmark = raw.get("benchmark") or {}
    benchmark_type = benchmark.get("type")
    return benchmark_type if isinstance(benchmark_type, str) else None


def generate_benchmark_rollup(
    logs_dir: Path,
    benchmark_type: str,
    scripts_dir: Path,
) -> bool:
    """Run benchmark-specific rollup.py to create benchmark-rollup.json."""
    rollup_script = scripts_dir / benchmark_type / "rollup.py"
    if not rollup_script.exists():
        print(f"No rollup script for benchmark type '{benchmark_type}' ({rollup_script})")
        return False

    print(f"Generating benchmark-rollup.json via {rollup_script.name} ({benchmark_type})...")
    try:
        result = subprocess.run(
            [sys.executable, str(rollup_script), str(logs_dir)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("ERROR: Rollup script timed out", file=sys.stderr)
        return False

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        print(f"ERROR: Rollup script failed (exit {result.returncode})", file=sys.stderr)
        return False

    rollup_path = logs_dir / "benchmark-rollup.json"
    if rollup_path.exists():
        print(f"Created {rollup_path}")
        return True

    print("WARNING: Rollup script completed but benchmark-rollup.json was not created", file=sys.stderr)
    return False


def resolve_job_paths(path: Path) -> tuple[Path, Path, str]:
    """Return (job_dir, logs_dir, job_id)."""
    path = path.resolve()

    if path.name == "logs":
        logs_dir = path
        job_dir = path.parent
    elif (path / "logs").is_dir():
        job_dir = path
        logs_dir = path / "logs"
    elif path.is_dir():
        job_dir = path.parent
        logs_dir = path
    else:
        raise FileNotFoundError(f"Job directory not found: {path}")

    if not logs_dir.is_dir():
        raise FileNotFoundError(f"Log directory not found: {logs_dir}")

    job_id = job_dir.name
    if not job_id.isdigit():
        raise ValueError(f"Could not infer SLURM job id from path (expected numeric dir name): {job_dir}")

    return job_dir, logs_dir, job_id


def copy_job_metadata_to_logs(job_dir: Path, logs_dir: Path, job_id: str) -> None:
    """Copy job metadata into logs/ so it is included in the S3 upload."""
    for name in ("config.yaml", "sbatch_script.sh", f"{job_id}.json", "git_state.txt"):
        src = job_dir / name
        if not src.exists():
            continue
        dst = logs_dir / name
        shutil.copy2(src, dst)
        print(f"Copied {name} -> {dst.relative_to(job_dir)}")


def build_s3_url(settings: S3Settings, date_str: str, job_id: str) -> str:
    prefix = settings.prefix.rstrip("/")
    return f"s3://{settings.bucket}/{prefix}/{date_str}/{job_id}/"


def resolve_credentials(settings: S3Settings) -> tuple[str | None, str | None]:
    access_key = os.path.expandvars(settings.access_key_id) if settings.access_key_id else None
    secret_key = os.path.expandvars(settings.secret_access_key) if settings.secret_access_key else None
    access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    return access_key, secret_key


def aws_cli_available() -> bool:
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def upload_with_aws_cli(
    logs_dir: Path,
    s3_url: str,
    settings: S3Settings,
    *,
    dry_run: bool,
) -> int:
    cmd = ["aws", "s3", "sync", str(logs_dir), s3_url]
    if settings.endpoint_url:
        cmd.extend(["--endpoint-url", settings.endpoint_url])
    if settings.region:
        cmd.extend(["--region", settings.region])
    if dry_run:
        cmd.append("--dryrun")

    env = os.environ.copy()
    access_key, secret_key = resolve_credentials(settings)
    if access_key:
        env["AWS_ACCESS_KEY_ID"] = access_key
    if secret_key:
        env["AWS_SECRET_ACCESS_KEY"] = secret_key
    if settings.region:
        env.setdefault("AWS_DEFAULT_REGION", settings.region)

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    return result.returncode


def upload_with_boto3(
    logs_dir: Path,
    s3_url: str,
    settings: S3Settings,
    *,
    dry_run: bool,
) -> int:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print(
            "Neither `aws` CLI nor boto3 is available.\n"
            "Install one of: pip install awscli  OR  pip install boto3",
            file=sys.stderr,
        )
        return 1

    # s3://bucket/prefix/date/job_id/
    without_scheme = s3_url.removeprefix("s3://")
    bucket, _, key_prefix = without_scheme.partition("/")
    key_prefix = key_prefix.rstrip("/") + "/"

    session_kwargs: dict[str, Any] = {}
    access_key, secret_key = resolve_credentials(settings)
    if access_key and secret_key:
        session_kwargs["aws_access_key_id"] = access_key
        session_kwargs["aws_secret_access_key"] = secret_key
    if settings.region:
        session_kwargs["region_name"] = settings.region

    client_kwargs: dict[str, Any] = {}
    if settings.endpoint_url:
        client_kwargs["endpoint_url"] = settings.endpoint_url

    client = boto3.client("s3", **session_kwargs, **client_kwargs)

    uploaded = 0
    skipped = 0
    errors = 0

    for local_path in sorted(logs_dir.rglob("*")):
        if not local_path.is_file():
            continue

        rel = local_path.relative_to(logs_dir).as_posix()
        key = f"{key_prefix}{rel}"
        local_size = local_path.stat().st_size

        try:
            head = client.head_object(Bucket=bucket, Key=key)
            remote_size = head["ContentLength"]
            if remote_size == local_size:
                skipped += 1
                continue
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                print(f"ERROR checking s3://{bucket}/{key}: {exc}", file=sys.stderr)
                errors += 1
                continue

        if dry_run:
            print(f"would upload {local_path} -> s3://{bucket}/{key}")
            uploaded += 1
            continue

        print(f"uploading {local_path} -> s3://{bucket}/{key}")
        client.upload_file(str(local_path), bucket, key)
        uploaded += 1

    print(f"Done: {uploaded} uploaded, {skipped} skipped, {errors} errors")
    return 1 if errors else 0


def count_local_files(logs_dir: Path) -> int:
    return sum(1 for path in logs_dir.rglob("*") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a srtctl job log directory to S3 (resume-safe via sync)."
    )
    parser.add_argument(
        "job_path",
        type=Path,
        help="Job output dir (outputs/JOBID) or log dir (outputs/JOBID/logs)",
    )
    parser.add_argument(
        "--cluster-config",
        type=Path,
        default=None,
        help="Path to srtslurm.yaml (default: auto-discover)",
    )
    parser.add_argument("--bucket", help="S3 bucket (overrides srtslurm.yaml reporting.s3.bucket)")
    parser.add_argument("--prefix", help="S3 prefix (default: srtslurm or reporting.s3.prefix)")
    parser.add_argument("--endpoint-url", help="Custom S3 endpoint URL")
    parser.add_argument("--region", help="AWS region")
    parser.add_argument(
        "--date",
        help="Upload date folder YYYY-MM-DD (default: today UTC, same as postprocess)",
    )
    parser.add_argument(
        "--skip-copy-config",
        action="store_true",
        help="Do not copy config.yaml, sbatch_script.sh, {job_id}.json, or git_state.txt into logs/ before upload",
    )
    parser.add_argument(
        "--benchmark-type",
        help="Benchmark type for rollup (default: read from job config.yaml)",
    )
    parser.add_argument(
        "--skip-rollup",
        action="store_true",
        help="Skip generating benchmark-rollup.json before upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without sending files",
    )
    parser.add_argument(
        "--force-boto3",
        action="store_true",
        help="Use boto3 instead of aws s3 sync",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        job_dir, logs_dir, job_id = resolve_job_paths(args.job_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    cluster_config_path = find_cluster_config(args.cluster_config)
    cluster_config = load_cluster_config(cluster_config_path)
    settings = load_s3_settings(cluster_config)

    if args.bucket:
        settings = S3Settings(
            bucket=args.bucket,
            prefix=args.prefix or (settings.prefix if settings else "srtslurm"),
            region=args.region or (settings.region if settings else None),
            endpoint_url=args.endpoint_url or (settings.endpoint_url if settings else None),
            access_key_id=settings.access_key_id if settings else None,
            secret_access_key=settings.secret_access_key if settings else None,
        )
    elif settings:
        if args.prefix:
            settings = S3Settings(
                bucket=settings.bucket,
                prefix=args.prefix,
                region=args.region or settings.region,
                endpoint_url=args.endpoint_url or settings.endpoint_url,
                access_key_id=settings.access_key_id,
                secret_access_key=settings.secret_access_key,
            )
        elif args.region or args.endpoint_url:
            settings = S3Settings(
                bucket=settings.bucket,
                prefix=settings.prefix,
                region=args.region or settings.region,
                endpoint_url=args.endpoint_url or settings.endpoint_url,
                access_key_id=settings.access_key_id,
                secret_access_key=settings.secret_access_key,
            )
    else:
        print(
            "ERROR: No S3 config found. Set reporting.s3 in srtslurm.yaml or pass --bucket.",
            file=sys.stderr,
        )
        return 1

    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    s3_url = build_s3_url(settings, date_str, job_id)

    print(f"Job ID:     {job_id}")
    print(f"Job dir:    {job_dir}")
    print(f"Logs dir:   {logs_dir} ({count_local_files(logs_dir)} files)")
    print(f"S3 dest:    {s3_url}")
    if cluster_config_path:
        print(f"Config:     {cluster_config_path}")
    print()

    if not args.skip_copy_config:
        copy_job_metadata_to_logs(job_dir, logs_dir, job_id)
        print()

    if not args.skip_rollup:
        benchmark_type = args.benchmark_type or load_benchmark_type(job_dir)
        if not benchmark_type:
            print("WARNING: Could not determine benchmark type; skipping rollup", file=sys.stderr)
        elif benchmark_type == "manual":
            print("Benchmark type is 'manual'; skipping rollup")
        else:
            scripts_dir = find_benchmark_scripts_dir(cluster_config)
            generate_benchmark_rollup(logs_dir, benchmark_type, scripts_dir)
            print()

    if args.force_boto3 or not aws_cli_available():
        if not args.force_boto3:
            print("aws CLI not found, falling back to boto3")
        return upload_with_boto3(logs_dir, s3_url, settings, dry_run=args.dry_run)

    return upload_with_aws_cli(logs_dir, s3_url, settings, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
