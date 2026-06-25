# -*- coding: utf-8 -*-
"""
下载真实疲劳数据 (第三方数据, 不随仓库分发, 用本脚本按需获取)
==============================================================
1. Virkler 裂纹扩展数据 (68 试件, 2024-T3 铝)
   来源: WarrRich/Virkler-Data (从 Bogdanoff & Kozin 1985 图4.5.3 数字化的近似版)
   底层实验: Virkler et al. 1977
2. NIMS 钢材疲劳强度数据 (437 条, 成分+热处理 -> 疲劳强度)
   来源: qq-shu/Fatigue-Dataset (取自 NIMS MatNavi, Agrawal et al. 2014)

注意: 这些是第三方数据, 各有来源/许可, 请遵守原始 license。
运行: python src/download_real_data.py
"""
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

FILES = {
    "virkler/VirklerData.csv":
        "https://raw.githubusercontent.com/WarrRich/Virkler-Data/main/VirklerData.csv",
    "virkler/README.md":
        "https://raw.githubusercontent.com/WarrRich/Virkler-Data/main/README.md",
    "virkler/LICENSE":
        "https://raw.githubusercontent.com/WarrRich/Virkler-Data/main/LICENSE",
    "nims/Fatigue-Dataset.csv":
        "https://raw.githubusercontent.com/qq-shu/Fatigue-Dataset/main/Fatigue-Dataset.csv",
    "nims/README.md":
        "https://raw.githubusercontent.com/qq-shu/Fatigue-Dataset/main/README.md",
}

for rel, url in FILES.items():
    dst = os.path.join(DATA, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dst)
        print(f"OK   {rel}  ({os.path.getsize(dst)} bytes)")
    except Exception as e:
        print(f"FAIL {rel}: {e}")

print("\n完成。Virkler -> data/virkler/, NIMS -> data/nims/")
print("接着可运行 src/virkler_real_data.py 探索 Virkler 数据。")
