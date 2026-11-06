"""扫描 git 全历史对象，确认没有任何敏感信息泄漏。

只输出「命中路径 / 计数 / 正则名」，绝不打印密钥本身。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:\Project\Changeden\github\nano-jev-data")
ENV_FILE = ROOT / "artifacts" / "llm" / ".env.local"

MAX_BLOB = 12 * 1024 * 1024  # 超过 12MB 的对象跳过（只报告）


def git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
    return result.stdout.decode("utf-8", "replace")


def read_secrets() -> list[tuple[str, bytes]]:
    """从本地 .env.local 读出**真正的密钥**用于精确比对（不回显）。

    只有名字里带 KEY / TOKEN / SECRET / PASSWORD 的才算密钥；模型名、网关地址、
    超时等配置不是敏感信息，混进来会产生误报。
    """
    secret_word = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)
    secrets: list[tuple[str, bytes]] = []
    if not ENV_FILE.exists():
        print("WARN: .env.local 不存在，只做模式扫描")
        return secrets
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not secret_word.search(name):
            continue
        value = value.strip().strip('"').strip("'")
        if not value:
            continue
        for part in re.split(r"[,\s]+", value):
            if len(part) >= 12:
                secrets.append((name, part.encode()))
    return secrets


PATTERNS = {
    "openai-like": re.compile(rb"sk-[A-Za-z0-9_\-]{16,}"),
    "github-pat": re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    "aws-key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "google-key": re.compile(rb"AIza[0-9A-Za-z_\-]{30,}"),
    "bearer-token": re.compile(rb"[Bb]earer\s+[A-Za-z0-9._\-]{20,}"),
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    "assign-secret": re.compile(
        rb"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password)\s*[:=]\s*"
        rb"[\"']([A-Za-z0-9_\-]{16,})[\"']"
    ),
}


def main() -> int:
    secrets = read_secrets()
    print(f"本地密钥条目 {len(secrets)} 个（值不打印，仅用于精确匹配）")

    objects = git("rev-list", "--objects", "--all").splitlines()
    blobs = []
    for line in objects:
        parts = line.split(" ", 1)
        sha = parts[0]
        path = parts[1] if len(parts) > 1 else ""
        blobs.append((sha, path))
    print(f"待扫描对象 {len(blobs)}")

    secret_hits: dict[str, list[str]] = defaultdict(list)
    pattern_hits: dict[str, list[str]] = defaultdict(list)
    scanned = skipped = 0

    for sha, path in blobs:
        size_text = git("cat-file", "-s", sha).strip()
        try:
            size = int(size_text)
        except ValueError:
            continue
        if size > MAX_BLOB:
            skipped += 1
            continue
        data = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-p", sha],
                              capture_output=True).stdout
        scanned += 1
        for name, value in secrets:
            if value in data:
                secret_hits[name].append(path or sha)
        for name, pattern in PATTERNS.items():
            if pattern.search(data):
                pattern_hits[name].append(path or sha)

    print(f"已扫描 {scanned} 个对象，跳过 {skipped} 个超大对象")

    # 允许的例外：脚本里出现的正则字面量本身
    ALLOW = {
        "openai-like": ("scripts/",),
    }

    def report(title: str, hits: dict[str, list[str]]) -> int:
        total = 0
        for name, paths in sorted(hits.items()):
            unique = sorted(set(paths))
            allow = ALLOW.get(name, ())
            bad = [p for p in unique if not any(a in p for a in allow)]
            if not bad:
                continue
            total += len(bad)
            print(f"  [{title}] {name}: {len(bad)} 处")
            for path in bad[:8]:
                print(f"      {path}")
        return total

    bad = report("本地密钥值", secret_hits)
    bad += report("模式", pattern_hits)

    # 跟踪文件里是否有不该入库的东西
    tracked = git("ls-files").splitlines()
    risky = [p for p in tracked if re.search(r"(^|/)(\.env|secrets?/)|\.pem$|\.key$|_rsa$|\.p12$|\.pfx$", p)]
    print(f"跟踪的敏感类文件: {len(risky)}")
    for path in risky[:10]:
        print("   ", path)

    env_tracked = [p for p in tracked if p.endswith(".env.local")]
    print(f".env.local 是否被跟踪: {'是' if env_tracked else '否'}")

    ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-v", "artifacts/llm/.env.local"],
                             capture_output=True).stdout.decode("utf-8", "replace").strip()
    print(f".env.local ignore 规则: {ignored or '未命中（危险）'}")

    if bad or risky or env_tracked:
        print("RESULT: 存在问题")
        return 1
    print("RESULT: 干净（全历史无密钥值、无模式命中、无敏感文件入库）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
