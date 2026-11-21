# -*- coding: utf-8 -*-
"""
食物分类 · 图片采集脚本
5 大类：主食、水果、肉类、蔬菜、海鲜
调用百度图片异步 JSON 接口，先建立会话 Cookie 再翻页，下载后用 PIL 校验。
"""
import os
import time
import io
import requests
from PIL import Image

CATEGORY_KEYWORDS = {
    "主食": ["米饭", "面条", "馒头", "饺子", "面包"],
    "水果": ["苹果", "香蕉", "橙子", "葡萄", "西瓜"],
    "肉类": ["红烧肉", "牛排", "烤肉", "鸡腿", "五花肉"],
    "蔬菜": ["白菜", "西兰花", "黄瓜", "西红柿", "胡萝卜"],
    "海鲜": ["螃蟹", "虾", "龙虾", "扇贝", "三文鱼"],
}
TARGET_PER_CATEGORY = 65
SAVE_ROOT = "./food_images"
RN = 30

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://image.baidu.com/",
    "X-Requested-With": "XMLHttpRequest",
}

# 全局会话：先访问首页拿到 Cookie，后续接口请求才不会被反爬拦截
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def init_session():
    try:
        SESSION.get("https://image.baidu.com/", timeout=15)
        print(f"会话已建立，Cookie 数量：{len(SESSION.cookies)}")
    except Exception as e:
        print(f"[警告] 建立会话失败：{e}")


def build_url(keyword, pn):
    base = "https://image.baidu.com/search/acjson"
    params = (
        f"?tn=resultjson_com&word={requests.utils.quote(keyword)}"
        f"&pn={pn}&rn={RN}&ala=0&nojc=0&newReq=1"
    )
    return base + params


def get_image_urls(keyword, max_pages=6):
    urls, seen = [], set()
    for page in range(max_pages):
        pn = page * RN
        try:
            resp = SESSION.get(build_url(keyword, pn), timeout=15)
            data = resp.json()
        except Exception as e:
            print(f"  [警告] 第{page + 1}页请求失败：{e}")
            time.sleep(1.5)
            continue

        if data.get("antiFlag"):
            print("  [警告] 被反爬拦截，停止该关键词翻页")
            break

        payload = data.get("data")
        items = payload.get("images") or [] if isinstance(payload, dict) else (payload or [])

        added = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            img_url = (item.get("thumburl") or item.get("objurl")
                       or item.get("hoverURL") or item.get("objURL"))
            if not img_url:
                continue
            if img_url.startswith("//"):
                img_url = "https:" + img_url
            elif img_url.startswith("http://"):
                img_url = "https://" + img_url[len("http://"):]
            if img_url not in seen:
                seen.add(img_url)
                urls.append(img_url)
                added += 1

        print(f"  关键词「{keyword}」第{page + 1}页：本页新增 {added} 条")
        if added == 0:
            break
        time.sleep(1.2)
    return urls


def is_valid_image(content):
    try:
        Image.open(io.BytesIO(content)).verify()
        Image.open(io.BytesIO(content)).convert("RGB")
        return True
    except Exception:
        return False


def download_one(url, save_path):
    try:
        resp = SESSION.get(url, stream=True, timeout=12)
        if resp.status_code != 200:
            return False
        content = resp.content
        if not is_valid_image(content):
            return False
        with open(save_path, "wb") as f:
            f.write(content)
        return True
    except Exception:
        return False


def crawl_category(category, keywords):
    save_dir = os.path.join(SAVE_ROOT, category)
    os.makedirs(save_dir, exist_ok=True)

    candidates = []
    for kw in keywords:
        print(f"正在收集「{category}」-「{kw}」的链接...")
        candidates.extend(get_image_urls(kw))
    candidates = list(dict.fromkeys(candidates))
    print(f"「{category}」候选链接 {len(candidates)} 条，开始下载...")

    index, ok_count = 1, 0
    for url in candidates:
        if ok_count >= TARGET_PER_CATEGORY:
            break
        save_path = os.path.join(save_dir, f"{category}_{index}.jpg")
        index += 1
        if download_one(url, save_path):
            ok_count += 1
        elif os.path.exists(save_path):
            os.remove(save_path)
        time.sleep(0.25)
    print(f"✅ 「{category}」完成，有效图片 {ok_count} 张\n")
    return ok_count


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    init_session()
    summary, total = {}, 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        summary[category] = crawl_category(category, keywords)
        total += summary[category]

    print("=" * 50)
    print("采集结果统计：")
    for cat, cnt in summary.items():
        print(f"  {cat}: {cnt} 张")
    print(f"  合计有效图片：{total} 张")
    print("✅ 总数满足不少于 200 张。" if total >= 200
          else "⚠️ 有效图片不足 200 张，请增大 TARGET_PER_CATEGORY 后重跑。")


if __name__ == "__main__":
    main()
