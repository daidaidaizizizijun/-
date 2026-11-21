# -*- coding: utf-8 -*-
"""诊断百度/必应图片接口可用性。"""
import requests

full_headers = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://image.baidu.com/search/index?tn=baiduimage&word=%E8%8B%B9%E6%9E%9C",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": '"Chromium";v="122", "Not(A:Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

print("===== 测试百度：先建 Session 再请求 =====")
sess = requests.Session()
sess.headers.update(full_headers)
try:
    home = sess.get("https://image.baidu.com/", timeout=15)
    print("首页状态码：", home.status_code, "| Cookie 数量：", len(sess.cookies))
    api = ("https://image.baidu.com/search/acjson?tn=resultjson_com"
           "&word=%E8%8B%B9%E6%9E%9C&pn=0&rn=30&ala=0&nojc=0&newReq=1")
    r = sess.get(api, timeout=15)
    print("接口状态码：", r.status_code)
    j = r.json()
    print("antiFlag：", j.get("antiFlag"), "| message：", j.get("message"))
    payload = j.get("data")
    n = len(payload.get("images", [])) if isinstance(payload, dict) else (len(payload) if payload else 0)
    print("返回图片条数：", n)
except Exception as e:
    print("百度测试异常：", e)

print("\n===== 测试必应图片 HTML 接口 =====")
try:
    bing_headers = {
        "User-Agent": full_headers["User-Agent"],
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    url = "https://www.bing.com/images/async?q=%E8%8B%B9%E6%9E%9C&first=1&count=20&mmasync=1"
    rb = requests.get(url, headers=bing_headers, timeout=15)
    print("必应状态码：", rb.status_code, "| 页面长度：", len(rb.text))
    # 必应结果里图片原图链接在 murl 属性
    import re
    murls = re.findall(r'murl&quot;:&quot;(.*?)&quot;', rb.text)
    if not murls:
        murls = re.findall(r'"murl":"(.*?)"', rb.text)
    print("解析到 murl 数量：", len(murls))
    for u in murls[:3]:
        print("  ", u)
except Exception as e:
    print("必应测试异常：", e)
