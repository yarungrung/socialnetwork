import os
import random
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

#先設定中文
# 讓所有 Matplotlib 畫出來的圖表直接內嵌顯示在 Notebook 儲存格下方
%matplotlib inline
# 處理中文標籤防包包、豆腐字設定
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]  # Windows 微軟正黑體
plt.rcParams["axes.unicode_minus"] = False  # 確保負號能正常顯示

#將資料匯入
from google.colab import drive
drive.mount('/content/drive')
data_folder = "/content/drive/MyDrive/社會網路"

output_folder = "final_output"
print("【系統提示】正在從 OSM 下載台中市道路網（這需要 1-2 分鐘，請耐心稍候）...")
G_raw = ox.graph_from_place("Taichung, Taiwan", network_type="drive")

G_proj = ox.project_graph(G_raw, to_crs="EPSG:3826")
G_undirected = G_proj.to_undirected()

print(f"▶ 【路網完成】成功載入台中市路網！")
print(f"   - 道路節點數: {G_undirected.number_of_nodes()}")
print(f"   - 道路線段數: {G_undirected.number_of_edges()}")
print(f"   - 當前座標系統: {G_proj.graph['crs']}")
print("-" * 50)

LIMIT_X = (180000, 250000)
LIMIT_Y = (2650000, 2710000)
data = {}

def clean_and_project(gdf, name):
    """【修正版】強迫指定初始座標為 WGS84 (經緯度)，再精準投影至 TWD97 (公尺)"""
    if gdf is None or len(gdf) == 0 or gdf.geometry.iloc[0] is None:
        return None

    # 💡 修正核心：不管三七二十一，CSV讀進來的一律先設定為 WGS84 經緯度
    # 如果原本就是 TWD97 (數值幾十萬)，我們用 try-except 來防範
    first_point = gdf.geometry.iloc[0]

    if first_point.x > 180:
        # 數值很大 (例如 217432)，代表本來就是 TWD97 投影座標
        gdf.crs = "EPSG:3826"
    else:
        # 數值很小 (例如 120.68)，代表是標準 WGS84 經緯度
        gdf.crs = "EPSG:4326"  # 先給它 WGS84 身分證
        gdf = gdf.to_crs("EPSG:3826")  # 再精準轉成 TWD97 公尺

    # 進行邊邊角角的範圍過濾
    gdf = gdf[
        (gdf.geometry.x >= LIMIT_X[0])
        & (gdf.geometry.x <= LIMIT_X[1])
        & (gdf.geometry.y >= LIMIT_Y[0])
        & (gdf.geometry.y <= LIMIT_Y[1])
    ]

    print(f"✅ 成功導入 【{name}】 -> 範圍內有效點數: {len(gdf)}")
    return gdf


def try_find_xy_columns(df):
    """輔助函式：自動尋找 Dataframe 裡的經緯度或 XY 座標欄位"""
    cols = df.columns
    x_col = next(
        (
            c
            for c in cols
            if "X" in c.upper()
            or "經" in c
            or "EAST" in c.upper()
            or "LONG" in c.upper()
        ),
        None,
    )
    y_col = next(
        (
            c
            for c in cols
            if "Y" in c.upper()
            or "緯" in c
            or "NORTH" in c.upper()
            or "LAT" in c.upper()
        ),
        None,
    )
    return x_col, y_col


print("【系統提示】開始依據實際檔案清單讀取點位...")

# 匯入各種點資料=================================================
# 1. 避難收容所 (CSV 檔)
# =====================================================================
try:
    shelter_file = "臺中市避難收容所位置及收容人數_CSV.csv"  # 依截圖檔名修正
    path = os.path.join(data_folder, shelter_file)
    print(f"DEBUG: Checking shelter path: {path}, exists: {os.path.exists(path)}")
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        x_col, y_col = try_find_xy_columns(df)
        df = df.dropna(subset=[x_col, y_col])
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[x_col], df[y_col])
        )
        data["shelter"] = clean_and_project(gdf, "避難收容所")
except Exception as e:
    print(f"❌ 避難收容所讀取失敗: {e}")

# =====================================================================
# 2. 台中量販店 (SHP 檔)
# =====================================================================
try:
    mart_file = "台中量販店.shp"
    path = os.path.join(data_folder, mart_file)
    print(f"DEBUG: Checking mart path: {path}, exists: {os.path.exists(path)}")
    if os.path.exists(path):
        gdf = gpd.read_file(path, encoding="cp950")
        data["mart"] = clean_and_project(gdf, "量販店/賣場")
except Exception as e:
    print(f"❌ 量販店讀取失敗: {e}")

# =====================================================================
# 3. 醫院 (SHP 檔)
# =====================================================================
try:
    hospital_file = "醫院.shp"
    path = os.path.join(data_folder, hospital_file)
    print(f"DEBUG: Checking hospital path: {path}, exists: {os.path.exists(path)}")
    if os.path.exists(path):
        gdf = gpd.read_file(path, encoding="cp950")
        data["hospital"] = clean_and_project(gdf, "醫院")
except Exception as e:
    print(f"❌ 醫院讀取失敗: {e}")

# =====================================================================
# 4. 五大超商 (Excel 檔)
# =====================================================================
try:
    store_file = "五大超商資料籍完整 (2).xlsx"
    path = os.path.join(data_folder, store_file)
    print(f"DEBUG: Checking store path: {path}, exists: {os.path.exists(path)}")
    if os.path.exists(path):
        df = pd.read_excel(path)
        x_col, y_col = try_find_xy_columns(df)
        df = df.dropna(subset=[x_col, y_col])
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[x_col], df[y_col])
        )
        data["store"] = clean_and_project(gdf, "五大超商")
except Exception as e:
    print(f"❌ 超商讀取失敗: {e}")

# =====================================================================
# 5. 加油站 (CSV 檔)
# =====================================================================
try:
    gas_file = "加油站.csv"
    path = os.path.join(data_folder, gas_file)
    print(f"DEBUG: Checking gas path: {path}, exists: {os.path.exists(path)}")
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        x_col, y_col = try_find_xy_columns(df)
        df = df.dropna(subset=[x_col, y_col])
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[x_col], df[y_col])
        )
        data["gas"] = clean_and_project(gdf, "加油站")
except Exception as e:
    print(f"❌ 加油站讀取失敗: {e}")

print("\n▶ 【全部讀取進度完成】目前收集到的機能圖層有:", list(data.keys()))


# 步驟 1：建立道路網節點的空間索引 (cKDTree)
# =====================================================================
print("【系統提示】正在建立道路網節點空間索引...")

# 抓出路網中所有節點的 ID 與座標
node_ids = list(G_undirected.nodes())
node_coords = np.array(
    [[G_undirected.nodes[n]["x"], G_undirected.nodes[n]["y"]] for n in node_ids]
)

# 使用 scipy 的 cKDTree 建立空間樹，加速最近鄰點搜尋
road_tree = cKDTree(node_coords)
# =====================================================================
# 步驟 2：定義空間對接與距離過濾函式 (距離小於 100 公尺限制)
# =====================================================================
def map_facilities_to_roads(gdf_facility, name, dist_limit=100.0):
    """將設施點位黏到最近的道路節點上，若幾何距離超過 dist_limit 則予以排除"""
    if gdf_facility is None or len(gdf_facility) == 0:
        print(f"⚠️ 找不到 【{name}】 的資料，跳過對接。")
        return {}

    # 抓出設施的座標
    facility_coords = np.array(
        [[geom.x, geom.y] for geom in gdf_facility.geometry]
    )

    # 查詢最近的道路節點 (distances: 距離, indices: 節點在 node_coords 中的索引)
    distances, indices = road_tree.query(facility_coords)

    mapped_results = {}
    valid_count = 0

    for idx, (dist, node_idx) in enumerate(zip(distances, indices)):
        # 💡 對齊妳們討論的邏輯：幾何距離必須在限制範圍內 (例如 100 公尺)
        if dist <= dist_limit:
            target_road_node = node_ids[node_idx]

            # 建立關聯：記錄這個道路節點上有哪些設施
            if target_road_node not in mapped_results:
                mapped_results[target_road_node] = []

            mapped_results[target_road_node].append(
                {
                    "facility_type": name,
                    "distance_to_road": dist,
                    "data_row": gdf_facility.iloc[idx].to_dict(),
                }
            )
            valid_count += 1

    print(
        f"▶ 【對接完成】{name} 共 {len(gdf_facility)} 點，成功黏上路網: {valid_count} 點 (臨界限制: {dist_limit}m)"
    )
    return mapped_results


# 批次執行所有設施圖層的對接
# =====================================================================
print("\n【系統提示】開始進行空間節點對接與距離篩選...")

# 用一個大字典裝「道路節點 ➔ 有哪些黏在上面的機能設施」
road_node_to_facilities = {}


def merge_mapping(new_mapping):
    for node, fac_list in new_mapping.items():
        if node not in road_node_to_facilities:
            road_node_to_facilities[node] = []
        road_node_to_facilities[node].extend(fac_list)


# 依序對接有讀成功的圖層 (限定幾何對接距離 100 公尺)
DISTANCE_LIMIT = 100.0

if "shelter" in data:
    merge_mapping(
        map_facilities_to_roads(
            data["shelter"], "避難收容所", dist_limit=DISTANCE_LIMIT
        )
    )
if "mart" in data:
    merge_mapping(
        map_facilities_to_roads(
            data["mart"], "量販店/賣場", dist_limit=DISTANCE_LIMIT
        )
    )
if "hospital" in data:
    merge_mapping(
        map_facilities_to_roads(
            data["hospital"], "醫院", dist_limit=DISTANCE_LIMIT
        )
    )
if "store" in data:
    merge_mapping(
        map_facilities_to_roads(
            data["store"], "五大超商", dist_limit=DISTANCE_LIMIT
        )
    )
if "gas" in data:
    merge_mapping(
        map_facilities_to_roads(
            data["gas"], "加油站", dist_limit=DISTANCE_LIMIT
        )
    )

print(
    f"\n🎉 都市路網空間對接全部搞定！全台中共有 {len(road_node_to_facilities)} 個道路節點成功綁定了防災機能點位。"

print("【系統提示】正在精確建立台中市基礎空間網格檔...")

# =====================================================================
# 設定網格大小 (妳手寫筆記上是 10 公尺)
# =====================================================================
# 💡 提示：交報告前請改成 10。如果現在測試想快一點，可以先設 100
GRID_SIZE = 100

# 1. 抓出目前台中路網上所有道路節點的 X, Y 座標極值（決定網格要鋪多大範圍）
node_xs = [G_undirected.nodes[n]["x"] for n in G_undirected.nodes()]
node_ys = [G_undirected.nodes[n]["y"] for n in G_undirected.nodes()]
minx, maxx = min(node_xs), max(node_xs)
miny, maxy = min(node_ys), max(node_ys)

# 2. 依據網格大小，在 X 軸與 Y 軸上切出等距離的坐標點
x_coords = np.arange(minx, maxx, GRID_SIZE)
y_coords = np.arange(miny, maxy, GRID_SIZE)

# 3. 利用雙重迴圈，把一個個 10m x 10m (或 100m x 100m) 的幾何正方形方塊(Polygon)生出來
print("正在生成網格幾何圖形...")
grid_geoms = [
    Polygon([(x, y), (x + GRID_SIZE, y), (x + GRID_SIZE, y + GRID_SIZE), (x, y + GRID_SIZE)])
    for x in x_coords
    for y in y_coords
]

# 4. 封裝成 GeoDataFrame，並精準指定台灣通用的 TWD97 座標系統 (EPSG:3826)
gdf_grids = gpd.GeoDataFrame(geometry=grid_geoms, crs="EPSG:3826")

# 5. 給每一個小格子一個獨立的身份證字號 (Grid_ID)
gdf_grids["Grid_ID"] = gdf_grids.index

print(f"\n🎉 【空白網格建置成功】")
print(f"   - 總共建立了 {len(gdf_grids)} 個規整空間網格單元。")
print(f"   - 網格座標系統已鎖定為: {gdf_grids.crs}")

