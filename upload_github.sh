#!/usr/bin/env bash
# 一键发布 ThesisForge 到 GitHub（需要先 gh auth login）
# 用法: bash upload_github.sh [仓库名]   # 默认 thesisforge
set -e
cd "$(dirname "$0")"

REPO_NAME="${1:-thesisforge}"
ZIP="dist/ThesisForge-v0.3.0-win64-offline.zip"
EXE="dist/ThesisForge-v0.3.0-win-x64.exe"

echo "== 0. 检查登录状态 =="
gh auth status >/dev/null 2>&1 || { echo "未登录 GitHub，请先运行: gh auth login"; exit 1; }

echo "== 1. 检查发布资产 =="
[ -f "$ZIP" ] || { echo "缺少 $ZIP，请先运行: python packaging/build_package.py"; exit 1; }
[ -f "$EXE" ] || { echo "缺少 $EXE，请先运行: python packaging/build_exe.py"; exit 1; }
ls -lh "$ZIP" "$EXE"

echo "== 2. 创建仓库并推送 =="
gh repo create "$REPO_NAME" --public --source . --remote origin --push \
  --description "AI 专业毕设一站式本地工作台：数据集/训练/实验对比/AI分析/学位论文生成" \
  || { echo "仓库可能已存在，尝试直接推送"; git remote add origin "https://github.com/$(gh api user -q .login)/$REPO_NAME.git" 2>/dev/null || true; git push -u origin main; }

echo "== 3. 创建 Release v0.3.0 并上传资产 =="
gh release create v0.3.0 "$ZIP" "$EXE" \
  --title "毕设工坊 ThesisForge v0.3.0（Windows 独立 EXE + 离线整合包）" \
  --notes-file packaging/release-notes-v0.3.0.md

echo "== 完成 =="
gh repo view --web 2>/dev/null || true
