# -*- coding: utf-8 -*-
"""用 Git Credential Manager 的 OAuth 凭据完成发布：建仓库 → 推送 → Release + 安装包上传。
token 只存在于进程内，绝不打印。"""
import json
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OWNER = "yihang56666-sketch"
REPO = "thesisforge"
EXE = ROOT / "dist" / "ThesisForge-v0.3.0-win-x64.exe"
ZIP = ROOT / "dist" / "ThesisForge-v0.3.0-win64-offline.zip"
NOTES = ROOT / "packaging" / "release-notes-v0.3.0.md"


def get_token() -> str:
    out = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True, cwd=str(ROOT), timeout=60,
    ).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line[len("password="):].strip()
    raise SystemExit("未取到 GitHub 凭据")


def main():
    token = get_token()
    h = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    api = httpx.Client(base_url=f"https://api.github.com", headers=h, timeout=60)

    # 0. 身份确认
    me = api.get("/user").json()
    print("登录账号:", me.get("login"))

    # 1. 创建仓库（已存在则跳过）
    r = api.post("/user/repos", json={
        "name": REPO,
        "description": "AI 专业毕设一站式本地工作台：数据集/训练/实验对比/AI分析/学位论文生成",
        "private": False,
        "has_wiki": False,
        "auto_init": False,
    })
    if r.status_code in (200, 201):
        print("仓库已创建:", r.json()["full_name"])
    elif r.status_code == 422 and "already exists" in r.text:
        print("仓库已存在，继续")
    else:
        print("建仓库失败:", r.status_code, r.text[:300])
        sys.exit(1)

    # 2. 推送代码
    full = f"{OWNER}/{REPO}"
    url = f"https://github.com/{full}.git"
    subprocess.run(["git", "remote", "remove", "origin"], capture_output=True, cwd=str(ROOT))
    subprocess.run(["git", "remote", "add", "origin", url], cwd=str(ROOT), check=True)
    print("推送 main ...")
    rp = subprocess.run(["git", "push", "-u", "origin", "main"], cwd=str(ROOT),
                        capture_output=True, text=True, timeout=300)
    print("push:", rp.returncode, (rp.stderr or "").strip().splitlines()[-1] if rp.stderr else "")
    if rp.returncode != 0:
        sys.exit(1)

    # 3. 创建 Release（已存在则复用）
    body = NOTES.read_text(encoding="utf-8")
    r = api.post(f"/repos/{full}/releases", json={
        "tag_name": "v0.3.0",
        "target_commitish": "main",
        "name": "毕设工坊 ThesisForge v0.3.0（Windows 独立 EXE + 离线整合包）",
        "body": body,
    })
    if r.status_code in (200, 201):
        print("Release 已创建:", r.json().get("html_url"))
    elif r.status_code == 422 and "already_exists" in r.text:
        r = api.get(f"/repos/{full}/releases/tags/v0.3.0")
        print("Release 已存在，复用")
    else:
        print("建 Release 失败:", r.status_code, r.text[:300])
        sys.exit(1)
    release = r.json()
    upload_url = release["upload_url"].split("{")[0]

    # 4. 上传发布资产（独立 EXE + 离线整合包）
    existing = api.get(f"/repos/{full}/releases/{release['id']}/assets").json()
    for asset in (EXE, ZIP):
        if not asset.exists():
            print(f"缺少资产 {asset.name}，请先构建。")
            sys.exit(1)
        if any(a["name"] == asset.name for a in existing):
            print("资产已存在，跳过上传:", asset.name)
            continue
        size = asset.stat().st_size
        print(f"上传 {asset.name}（{size/1048576:.0f} MB）...")
        content_type = "application/octet-stream"
        if asset.suffix == ".zip":
            content_type = "application/zip"
        with open(asset, "rb") as f:
            ru = httpx.post(
                upload_url,
                params={"name": asset.name},
                headers={"Authorization": f"token {token}", "Content-Type": content_type},
                content=f, timeout=1800,
            )
        if ru.status_code in (200, 201):
            print("资产上传完成:", ru.json().get("browser_download_url"))
        else:
            print("资产上传失败:", ru.status_code, ru.text[:300])
            sys.exit(1)

    print("\n发布完成!")
    print(f"仓库: https://github.com/{full}")
    print(f"Release: {release.get('html_url')}")


if __name__ == "__main__":
    main()
