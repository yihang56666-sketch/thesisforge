"""Browser smoke test for the 10-step guided thesis wizard (Sprint 1)."""

import json
import re
import sys
import urllib.request

from playwright.sync_api import sync_playwright


BASE = "http://127.0.0.1:8765"
errors = []
PAGE_REF = {"page": None, "browser": None}


def reset_wizard_api():
    body = {
        "version": 1,
        "project": {"title": "", "direction": "", "author": "", "advisor": "", "goal": ""},
        "steps": {
            str(i): {"state": "todo", "completed": False, "saved_at": None}
            for i in range(1, 11)
        },
        "active_step": 1,
        "updated_at": "",
    }
    req = urllib.request.Request(
        f"{BASE}/api/wizard",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ensure_builtin_absent(key):
    with urllib.request.urlopen(f"{BASE}/api/datasets", timeout=10) as resp:
        current = json.loads(resp.read().decode("utf-8"))
    if any(d["id"] == key for d in current["datasets"]):
        req = urllib.request.Request(f"{BASE}/api/datasets/{key}", method="DELETE")
        with urllib.request.urlopen(req, timeout=10) as resp:
            json.loads(resp.read().decode("utf-8"))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"PASS: {msg}")


def main():
    reset_wizard_api()
    ensure_builtin_absent("wine")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        PAGE_REF["page"] = page
        PAGE_REF["browser"] = browser
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"PAGEERROR: {e}"))

        page.goto(f"{BASE}/#/wizard", wait_until="domcontentloaded")
        page.wait_for_selector(".wizard-page", timeout=15000)
        page.wait_for_load_state("networkidle")

        step_title = page.locator(".wizard-main h1").inner_text()
        check(step_title == "立项", f"第 1 步标题正确: {step_title}")
        chips = page.locator(".wizard-step-chip").count()
        check(chips == 10, f"10 个步骤芯片: {chips}")
        page.screenshot(path="scripts/_wizard_step1.png", full_page=True)

        page.fill('[data-wfield="title"]', "基于卷积神经网络的医学图像分类研究与优化")
        page.select_option('[data-wfield="direction"]', "图像分类")
        page.fill('[data-wfield="author"]', "张三")
        page.click('[data-wstep="done"]')
        page.wait_for_function(
            "document.querySelector('.wizard-main h1') && document.querySelector('.wizard-main h1').textContent.includes('数据')",
            timeout=15000,
        )
        check(page.locator(".wizard-main h1").inner_text() == "数据", "保存并继续后进入第 2 步")
        check(
            page.locator(".wizard-step-chip.active").inner_text().strip().startswith("2"),
            "步骤条高亮第 2 步",
        )

        before_text = page.locator(".wizard-panel").inner_text()
        before_n = int(re.search(r"当前已载入 (\d+) 个数据集", before_text).group(1))
        wine_card = page.locator('[data-builtin="wine"]')
        check(wine_card.count() == 1, "内置数据集卡片存在")
        wine_card.click()
        page.wait_for_selector(
            "table.data tbody tr:has-text('葡萄酒 Wine')",
            timeout=30000,
        )
        page.wait_for_load_state("networkidle")
        after_text = page.locator(".wizard-panel").inner_text()
        after_n = int(re.search(r"当前已载入 (\d+) 个数据集", after_text).group(1))
        check(
            after_n > before_n,
            f"数据集数量从 {before_n} 增加到 {after_n}",
        )
        check(
            "葡萄酒 Wine" in after_text or "葡萄酒 Wine（多分类）" in after_text,
            "载入 Wine 后数据集列表出现对应行",
        )
        page.screenshot(path="scripts/_wizard_step2.png", full_page=True)

        page.evaluate("location.hash = '#/dashboard'")
        page.wait_for_timeout(800)
        page.evaluate("location.hash = '#/wizard'")
        page.wait_for_selector(".wizard-page", timeout=15000)
        page.wait_for_load_state("networkidle")
        h1 = page.locator(".wizard-main h1").inner_text()
        rail = page.locator(".wizard-rail").inner_text()
        check(h1 == "数据", "切换页面后仍停留在第 2 步")
        check("基于卷积神经网络的医学图像分类研究与优化" in rail, "项目题目跨页面持久化")

        if errors:
            print("CONSOLE_ERRORS:", json.dumps(errors, ensure_ascii=False))
            raise AssertionError("页面出现 console/page error")
        print("ALL_WIZARD_CHECKS_PASSED")
        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        page = PAGE_REF.get("page")
        if page:
            try:
                print("PAGE_H1:", page.locator(".wizard-main h1").inner_text())
                print("PAGE_TOASTS:", page.locator("#toasts").inner_text())
                print("PAGE_BODY:", page.locator("body").inner_text()[:1500])
                page.screenshot(path="scripts/_wizard_fail.png", full_page=True)
            except Exception as e2:
                print("DIAG_ERR:", e2)
        print("CONSOLE_ERRORS:", json.dumps(errors, ensure_ascii=False), file=sys.stderr)
        print("FAIL:", exc, file=sys.stderr)
        sys.exit(1)
