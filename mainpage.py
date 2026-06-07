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
)
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

#開始跑自選經緯度的節點失能
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

print("【災害模擬準備】正在建立可重複使用的災後重算函式與災前基準分數...")

# =====================================================================
# 0. 參數設定
# =====================================================================
N_SIMULATIONS = 100     #先跑三次，記得改回去100
DISASTER_RADIUS = 100      # 公尺，道路失能半徑
DISTANCE_THRESHOLD = 3000  # 沿用災前：資源點 3 公里內形成生活圈連結
CORRIDOR_BUFFER = 250      # 沿用災前：生活圈道路廊道 buffer 寬度
RANDOM_SEED = 20260606     # 想每次得到不同結果可改成 None

RESOURCE_TYPES = ["醫院", "五大超商", "量販店", "加油站"]
FACTOR_FIELDS = ["醫院_因子分數", "五大超商_因子分數", "量販店_因子分數", "加油站_因子分數", "避難收容所_因子分數"]
BASELINE_SCORE_FIELD = "災前_網格防災韌性綜合分數"
POST_SCORE_FIELD = "災後_網格防災韌性綜合分數"
FINAL_SCORE_FIELD = "最終韌性變化分數"

rng = random.Random(RANDOM_SEED)

# =====================================================================
# 1. 建立災前基準：使用「跨區 Closeness + 五類資源總分」作為比較分數
# =====================================================================
required_baseline_fields = ["跨區_Closeness_權重", "生活圈防災機能總分數"]
missing = [c for c in required_baseline_fields if c not in gdf_grids_results.columns]
if missing:
    raise KeyError(f"找不到災前欄位：{missing}，請先執行前面的災前計分儲存格。")

gdf_grids_baseline = gdf_grids_results.copy()
for field in required_baseline_fields:
    gdf_grids_baseline[field] = gdf_grids_baseline[field].fillna(0.0)
gdf_grids_baseline[BASELINE_SCORE_FIELD] = (
    gdf_grids_baseline["跨區_Closeness_權重"] + gdf_grids_baseline["生活圈防災機能總分數"]
)

# =====================================================================
# 2. 預先整理資源節點與原始候選連線，避免 100 次模擬都從零開始配對
# =====================================================================
facility_records_base = []
for fac_id in G_fac_net.nodes():
    nd = G_fac_net.nodes[fac_id]
    raw = nd.get("raw_data", {})
    facility_records_base.append({
        "fac_id": fac_id,
        "facility_type": nd.get("facility_type"),
        "road_node": nd.get("road_node"),
        "indoor_capacity": float(raw.get("室內人數", 0) or 0)
    })

df_facility_base = pd.DataFrame(facility_records_base)
baseline_fac_edges = [(u, v) for u, v in G_fac_net.edges()]

if "global_counts" not in globals():
    global_counts = df_facility_base.groupby("facility_type").size().to_dict()

# =====================================================================
# 3. 幾何與路網工具函式
# =====================================================================
def _edge_geometry(G, u, v, data):
    if data is not None and "geometry" in data:
        return data["geometry"]
    u_d = G.nodes[u]
    v_d = G.nodes[v]
    return LineString([(u_d["x"], u_d["y"]), (v_d["x"], v_d["y"])])


def random_point_on_road(G, rng):
    """依道路長度加權，隨機在路網線段上取一個災害中心點。"""
    edge_items = []
    lengths = []
    iterator = G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    for item in iterator:
        if G.is_multigraph():
            u, v, k, data = item
        else:
            u, v, data = item
            k = None
        length = float(data.get("length", 0) or 0)
        if length <= 0:
            length = float(_edge_geometry(G, u, v, data).length)
        if length > 0:
            edge_items.append((u, v, k, data))
            lengths.append(length)
    if not edge_items:
        raise ValueError("路網沒有可抽樣的道路邊。")

    u, v, k, data = rng.choices(edge_items, weights=lengths, k=1)[0]
    geom = _edge_geometry(G, u, v, data)
    return geom.interpolate(rng.random() * geom.length), (u, v, k)


def remove_failed_roads(G, disaster_point, radius=100):
    """移除災害中心 radius 公尺內受影響的道路，代表道路失能。"""
    G_damaged = G.copy()
    impact_area = disaster_point.buffer(radius)
    edges_to_remove = []

    iterator = G_damaged.edges(keys=True, data=True) if G_damaged.is_multigraph() else G_damaged.edges(data=True)
    for item in iterator:
        if G_damaged.is_multigraph():
            u, v, k, data = item
        else:
            u, v, data = item
            k = None
        if _edge_geometry(G_damaged, u, v, data).intersects(impact_area):
            edges_to_remove.append((u, v, k))

    if G_damaged.is_multigraph():
        G_damaged.remove_edges_from(edges_to_remove)
    else:
        G_damaged.remove_edges_from([(u, v) for u, v, _ in edges_to_remove])

    return G_damaged, impact_area, len(edges_to_remove)


def louvain_partition_graph(G):
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_edges() == 0:
        return {n: i for i, n in enumerate(G.nodes())}
    if community_louvain is not None:
        return community_louvain.best_partition(G, weight="weight", random_state=RANDOM_SEED)
    comms = nx.community.louvain_communities(G, weight="weight", seed=RANDOM_SEED)
    return {n: cid for cid, nodes in enumerate(comms) for n in nodes}


def rebuild_facility_network_after_disaster(G_damaged):
    """沿用災前 G_fac_net 的候選連線；道路失能後若仍在 3 公里內可達才保留。"""
    G_post = nx.Graph()
    for row in facility_records_base:
        G_post.add_node(
            int(row["fac_id"]),
            facility_type=row["facility_type"],
            road_node=row["road_node"],
            indoor_capacity=row["indoor_capacity"]
        )

    dist_cache = {}
    for u, v in baseline_fac_edges:
        road_u = G_post.nodes[u]["road_node"]
        road_v = G_post.nodes[v]["road_node"]
        if road_u == road_v:
            if road_u in G_damaged:
                G_post.add_edge(u, v, weight=50.0)
            continue
        if road_u not in G_damaged or road_v not in G_damaged:
            continue
        if road_u not in dist_cache:
            try:
                dist_cache[road_u] = nx.single_source_dijkstra_path_length(
                    G_damaged, road_u, cutoff=DISTANCE_THRESHOLD, weight="length"
                )
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                dist_cache[road_u] = {}
        dist = dist_cache[road_u].get(road_v)
        if dist is not None and dist <= DISTANCE_THRESHOLD:
            G_post.add_edge(u, v, weight=float(dist))

    partition = louvain_partition_graph(G_post)
    nx.set_node_attributes(G_post, partition, "cluster_id")
    return G_post, partition


def build_post_cluster_layers(G_damaged, G_post):
    """建立災後生活圈中心、跨區 closeness，以及道路廊道 polygon。"""
    fac_rows = []
    for fac_id, nd in G_post.nodes(data=True):
        road_node = nd["road_node"]
        if road_node not in G_damaged:
            continue
        fac_rows.append({
            "fac_id": fac_id,
            "cluster_id": int(nd.get("cluster_id", -1)),
            "road_node": road_node,
            "facility_type": nd.get("facility_type"),
            "indoor_capacity": float(nd.get("indoor_capacity", 0) or 0),
            "x": G_damaged.nodes[road_node]["x"],
            "y": G_damaged.nodes[road_node]["y"]
        })
    df_fac_post = pd.DataFrame(fac_rows)
    if df_fac_post.empty:
        return gpd.GeoDataFrame(columns=["cluster_id", "生活圈跨區_Closeness", "geometry"], crs="EPSG:3826"), df_fac_post

    cluster_centroids_post = df_fac_post.groupby("cluster_id")[["x", "y"]].mean().reset_index()
    G_inter_post = nx.Graph()
    cluster_to_road = {}
    for _, row in cluster_centroids_post.iterrows():
        cid = int(row["cluster_id"])
        G_inter_post.add_node(cid, x=row["x"], y=row["y"])
        try:
            cluster_to_road[cid] = ox.distance.nearest_nodes(G_damaged, X=row["x"], Y=row["y"])
        except Exception:
            cluster_to_road[cid] = None

    cids = list(G_inter_post.nodes())
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            c_i, c_j = cids[i], cids[j]
            r_i, r_j = cluster_to_road.get(c_i), cluster_to_road.get(c_j)
            if r_i is None or r_j is None:
                continue
            try:
                dist = nx.shortest_path_length(G_damaged, source=r_i, target=r_j, weight="length")
                G_inter_post.add_edge(c_i, c_j, weight=float(dist))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    inter_closeness_post = nx.closeness_centrality(G_inter_post, distance="weight") if G_inter_post.number_of_nodes() else {}
    cluster_centroids_post["生活圈跨區_Closeness"] = cluster_centroids_post["cluster_id"].map(inter_closeness_post).fillna(0.0)

    road_node_to_cluster_post = df_fac_post.set_index("road_node")["cluster_id"].to_dict()
    cluster_road_geoms = []
    iterator = G_damaged.edges(keys=True, data=True) if G_damaged.is_multigraph() else G_damaged.edges(data=True)
    for item in iterator:
        if G_damaged.is_multigraph():
            u, v, k, data = item
        else:
            u, v, data = item
        cu = road_node_to_cluster_post.get(u, -1)
        cv = road_node_to_cluster_post.get(v, -1)
        chosen = -1
        if cu == cv and cu != -1:
            chosen = cu
        elif cu != -1 and cv == -1:
            chosen = cu
        elif cv != -1 and cu == -1:
            chosen = cv
        if chosen != -1:
            cluster_road_geoms.append({"cluster_id": int(chosen), "geometry": _edge_geometry(G_damaged, u, v, data)})

    if not cluster_road_geoms:
        return gpd.GeoDataFrame(columns=["cluster_id", "生活圈跨區_Closeness", "geometry"], crs="EPSG:3826"), df_fac_post

    gdf_roads_post = gpd.GeoDataFrame(cluster_road_geoms, crs="EPSG:3826")
    gdf_polygons_post = gdf_roads_post.dissolve(by="cluster_id")
    gdf_polygons_post["geometry"] = gdf_polygons_post.geometry.buffer(CORRIDOR_BUFFER)
    gdf_polygons_post = gdf_polygons_post.reset_index()
    return gdf_polygons_post.merge(
        cluster_centroids_post[["cluster_id", "生活圈跨區_Closeness"]], on="cluster_id", how="left"
    ).fillna({"生活圈跨區_Closeness": 0.0}), df_fac_post


def score_post_clusters(gdf_polygons_post, df_fac_post):
    if gdf_polygons_post.empty:
        return gdf_polygons_post

    gdf_pop = gdf_v_pop.to_crs("EPSG:3826") if gdf_v_pop.crs != "EPSG:3826" else gdf_v_pop.copy()
    if "村里總面積" not in gdf_pop.columns:
        gdf_pop["村里總面積"] = gdf_pop.geometry.area

    intersections = gpd.overlay(gdf_polygons_post, gdf_pop, how="intersection")
    if intersections.empty:
        df_cluster_pop = pd.DataFrame({"cluster_id": gdf_polygons_post["cluster_id"], "生活圈真實總人口_分母": 0.0})
    else:
        intersections["碎片交集面積"] = intersections.geometry.area
        intersections["碎片分配人口"] = intersections["total"] * (intersections["碎片交集面積"] / intersections["村里總面積"])
        df_cluster_pop = intersections.groupby("cluster_id")["碎片分配人口"].sum().reset_index()
        df_cluster_pop.columns = ["cluster_id", "生活圈真實總人口_分母"]

    df_counts = df_fac_post.groupby(["cluster_id", "facility_type"]).size().unstack(fill_value=0).reset_index()
    df_indoor = df_fac_post[df_fac_post["facility_type"] == "避難收容所"].groupby("cluster_id")["indoor_capacity"].sum().reset_index()
    df_indoor.columns = ["cluster_id", "生活圈總室內人數_分子"]
    df_scores = df_counts.merge(df_indoor, on="cluster_id", how="left").merge(df_cluster_pop, on="cluster_id", how="left").fillna(0)

    for col in RESOURCE_TYPES:
        denom = float(global_counts.get(col, 0) or 0)
        df_scores[f"{col}_因子分數"] = (df_scores[col] / denom) if (col in df_scores.columns and denom > 0) else 0.0
    df_scores["避難收容所_因子分數"] = np.where(
        df_scores["生活圈真實總人口_分母"] > 0,
        df_scores["生活圈總室內人數_分子"] / df_scores["生活圈真實總人口_分母"],
        0.0
    )
    df_scores["生活圈防災機能總分數"] = df_scores[FACTOR_FIELDS].sum(axis=1)

    out = gdf_polygons_post.merge(df_scores[["cluster_id"] + FACTOR_FIELDS + ["生活圈防災機能總分數"]], on="cluster_id", how="left")
    for col in ["生活圈跨區_Closeness", "生活圈防災機能總分數"] + FACTOR_FIELDS:
        out[col] = out[col].fillna(0.0)
    out[POST_SCORE_FIELD] = out["生活圈跨區_Closeness"] + out["生活圈防災機能總分數"]
    return out


def assign_post_scores_to_grids(gdf_cluster_scores):
    if gdf_cluster_scores.empty:
        out = gdf_grids[["Grid_ID", "geometry"]].copy()
        out["cluster_id"] = -1
        out[POST_SCORE_FIELD] = 0.0
        return out

    join_cols = ["cluster_id", "生活圈跨區_Closeness", "生活圈防災機能總分數", POST_SCORE_FIELD, "geometry"]
    joined = gpd.sjoin(gdf_grids[["Grid_ID", "geometry"]], gdf_cluster_scores[join_cols], how="left", predicate="intersects")
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])
    df_unique = joined.groupby("Grid_ID").agg({
        "cluster_id": "first",
        "生活圈跨區_Closeness": "max",
        "生活圈防災機能總分數": "max",
        POST_SCORE_FIELD: "max"
    }).reset_index()

    out = gdf_grids[["Grid_ID", "geometry"]].merge(df_unique, on="Grid_ID", how="left")
    out["cluster_id"] = out["cluster_id"].fillna(-1).astype(int)
    out[POST_SCORE_FIELD] = out[POST_SCORE_FIELD].fillna(0.0)
    return out[["Grid_ID", "cluster_id", POST_SCORE_FIELD, "geometry"]]


def run_one_disaster_simulation(sim_id, rng):
    disaster_point, sampled_edge = random_point_on_road(G_undirected, rng)
    G_damaged, impact_area, removed_edges = remove_failed_roads(G_undirected, disaster_point, DISASTER_RADIUS)
    G_post, partition = rebuild_facility_network_after_disaster(G_damaged)
    gdf_post_polygons, df_fac_post = build_post_cluster_layers(G_damaged, G_post)
    gdf_post_scores = score_post_clusters(gdf_post_polygons, df_fac_post)
    gdf_grid_post = assign_post_scores_to_grids(gdf_post_scores)
    meta = {
        "simulation_id": sim_id,
        "災害點_X": disaster_point.x,
        "災害點_Y": disaster_point.y,
        "失能道路邊數": removed_edges,
        "災後生活圈數": len(set(partition.values())) if partition else 0,
        "災後平均網格分數": float(gdf_grid_post[POST_SCORE_FIELD].mean())
    }
    return gdf_grid_post[["Grid_ID", POST_SCORE_FIELD]], meta

print("✅ 災害模擬函式準備完成。接著執行下一格，就會開始模擬自選災害點。")

# 1. 先強迫卸載可能衝突的錯誤套件
!pip uninstall -y community python-louvain

# 2. 精準安裝含有 best_partition 的正確套件
!pip install python-louvain

import time

print(f"【災害模擬啟動】開始執行 {N_SIMULATIONS} 次隨機道路失能模擬，每次影響半徑 {DISASTER_RADIUS} 公尺...")

# 41 萬格 x 100 次若全部存成寬表會非常吃記憶體；這裡採用逐次加總，最後再除以次數。
post_score_sum = pd.Series(0.0, index=gdf_grids["Grid_ID"].values)
simulation_logs = []
start_time = time.time()

for sim_id in range(1, N_SIMULATIONS + 1):
    sim_scores, sim_meta = run_one_disaster_simulation(sim_id, rng)
    sim_series = sim_scores.set_index("Grid_ID")[POST_SCORE_FIELD].reindex(post_score_sum.index).fillna(0.0)
    post_score_sum = post_score_sum.add(sim_series, fill_value=0.0)
    simulation_logs.append(sim_meta)

    if sim_id == 1 or sim_id % 10 == 0 or sim_id == N_SIMULATIONS:
        elapsed = (time.time() - start_time) / 60
        print(
            f"   - 已完成 {sim_id:>3}/{N_SIMULATIONS} 次，"
            f"本次失能道路 {sim_meta['失能道路邊數']} 條，"
            f"災後生活圈 {sim_meta['災後生活圈數']} 個，"
            f"累計耗時 {elapsed:.1f} 分鐘"
        )

print("【結果彙整】正在合併 100 次災後網格分數並計算平均...")
df_post_avg = pd.DataFrame({
    "Grid_ID": post_score_sum.index,
    "災後100次平均分數": (post_score_sum.values / N_SIMULATIONS)
})

base_cols = gdf_grids_baseline[["Grid_ID", BASELINE_SCORE_FIELD, "geometry"]].copy()
gdf_resilience_result = base_cols.merge(df_post_avg[["Grid_ID", "災後100次平均分數"]], on="Grid_ID", how="left")
gdf_resilience_result["災後100次平均分數"] = gdf_resilience_result["災後100次平均分數"].fillna(0.0)
gdf_resilience_result[FINAL_SCORE_FIELD] = (
    gdf_resilience_result["災後100次平均分數"] - gdf_resilience_result[BASELINE_SCORE_FIELD]
)
gdf_resilience_result = gpd.GeoDataFrame(gdf_resilience_result, geometry="geometry", crs=gdf_grids.crs)
df_simulation_log = pd.DataFrame(simulation_logs)

print("\n🎉 【災害模擬完成】")
print(f"   👉 已完成 {N_SIMULATIONS} 次隨機災害模擬。")
print(f"   👉 變數 gdf_resilience_result 已包含 {FINAL_SCORE_FIELD}。")
print("   👉 變數 df_simulation_log 已紀錄每次災害位置、失能道路數與生活圈數。")

display(df_simulation_log.head())
display(gdf_resilience_result[["Grid_ID", BASELINE_SCORE_FIELD, "災後100次平均分數", FINAL_SCORE_FIELD]].sort_values(by=FINAL_SCORE_FIELD).head(10))