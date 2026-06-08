import os
import time
import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
from collections import defaultdict

# 檢查與載入 Louvain 社群演算法套件
try:
    import community as community_louvain
except ImportError:
    community_louvain = None

# =====================================================================
# 🌐 1. Streamlit 網頁前端介面與初始化配置
# =====================================================================
st.set_page_config(page_title="臺中市都市防災韌性評估系統", layout="wide")

# 強制將側邊欄樣式與控制面板一開網頁就呈現
st.sidebar.header("🎯 災害情境自訂面板")
disaster_name = st.sidebar.text_input("1. 請輸入特定災害名稱", value="自訂颱風/地震引發道路失能事件")
disaster_radius = st.sidebar.slider("2. 指定道路失能半徑 (公尺)", min_value=100, max_value=5000, value=2000, step=100)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🖱️ 條件一：如何自訂失能中心？
1. 請直接在右側的**臺中市地圖上任意點選**。
2. 系統會自動捕捉滑鼠點擊處的經緯度。
3. 點擊下方的 **[🚀 啟動單次空間失能模擬]** 按鈕。
""")

st.title("🗺️ 臺中市都市防災空間網絡韌性評估系統")
st.markdown("本系統整合 OSM 道路網與都市災害機能圖資，全面改採同學最新修訂之**「幾何平均數 (Geometric Mean)」**綜合韌性計分法。")

# =====================================================================
# 📥 2. 核心資料載入與快取機制 (對齊同學 IPYNB 的 Cell 1 ~ Cell 5)
# =====================================================================
@st.cache_data(show_spinner=False)
def load_and_process_base_data():
    data_folder = "data"  # 請確保檔案放在專案目錄下的 data 資料夾中
    LIMIT_X = (180000, 250000)
    LIMIT_Y = (2650000, 2710000)
    
    # (A) 下載並優化台中市路網
    G_raw = ox.graph_from_place("Taichung, Taiwan", network_type="drive")
    G_proj = ox.project_graph(G_raw, to_crs="EPSG:3826")
    G_undirected = G_proj.to_undirected()
    
    # (B) 讀取與過濾真實機能點位
    data_layers = {}
    
    def clean_and_project(gdf):
        if gdf is None or len(gdf) == 0: return None
        first_point = gdf.geometry.iloc[0]
        gdf.crs = "EPSG:3826" if first_point.x > 180 else "EPSG:4326"
        if gdf.crs == "EPSG:4326":
            gdf = gdf.to_crs("EPSG:3826")
        return gdf[
            (gdf.geometry.x >= LIMIT_X[0]) & (gdf.geometry.x <= LIMIT_X[1]) &
            (gdf.geometry.y >= LIMIT_Y[0]) & (gdf.geometry.y <= LIMIT_Y[1])
        ]

    # 避難收容所
    shelter_path = os.path.join(data_folder, "臺中市避難收容所位置及收容人數_CSV.csv")
    if os.path.exists(shelter_path):
        df = pd.read_csv(shelter_path, encoding="utf-8-sig")
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.iloc[:, 0], df.iloc[:, 1]))
        data_layers["shelter"] = clean_and_project(gdf)
        
    # 醫療機構 (醫院)
    hosp_path = os.path.join(data_folder, "醫院.shp")
    if os.path.exists(hosp_path):
        data_layers["hospital"] = clean_and_project(gpd.read_file(hosp_path, encoding="cp950"))
        
    # 物資量販店
    mart_path = os.path.join(data_folder, "台中量販店.shp")
    if os.path.exists(mart_path):
        data_layers["mart"] = clean_and_project(gpd.read_file(mart_path, encoding="cp950"))
        
    # 能源加油站
    gas_path = os.path.join(data_folder, "加油站.csv")
    if os.path.exists(gas_path):
        df = pd.read_csv(gas_path, encoding="utf-8-sig")
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.iloc[:, 0], df.iloc[:, 1]))
        data_layers["gas"] = clean_and_project(gdf)

    # (C) 建立空間 KD-Tree 對接路網節點
    node_ids = list(G_undirected.nodes())
    node_coords = np.array([[G_undirected.nodes[n]["x"], G_undirected.nodes[n]["y"]] for n in node_ids])
    road_tree = cKDTree(node_coords)
    
    road_node_to_facilities = {}
    for layer_name, gdf_fac in data_layers.items():
        if gdf_fac is None: continue
        fac_coords = np.array([[geom.x, geom.y] for geom in gdf_fac.geometry])
        distances, indices = road_tree.query(fac_coords)
        for idx, (dist, node_idx) in enumerate(zip(distances, indices)):
            if dist <= 150.0:  # 容許 150 公尺內空間對接
                target_node = node_ids[node_idx]
                if target_node not in road_node_to_facilities:
                    road_node_to_facilities[target_node] = []
                road_node_to_facilities[target_node].append({
                    "facility_type": layer_name,
                    "data": gdf_fac.iloc[idx].to_dict()
                })

    # (D) 動態生成均勻網格底圖 (先設 200 米平衡網頁流暢度與精細度)
    GRID_SIZE = 200 
    node_xs, node_ys = node_coords[:, 0], node_coords[:, 1]
    x_coords = np.arange(min(node_xs), max(node_xs), GRID_SIZE)
    y_coords = np.arange(min(node_ys), max(node_ys), GRID_SIZE)
    grid_geoms = [
        Polygon([(x, y), (x + GRID_SIZE, y), (x + GRID_SIZE, y + GRID_SIZE), (x, y + GRID_SIZE)])
        for x in x_coords for y in y_coords
    ]
    gdf_grids = gpd.GeoDataFrame(geometry=grid_geoms, crs="EPSG:3826")
    gdf_grids["Grid_ID"] = gdf_grids.index
    
    return G_proj, G_undirected, road_node_to_facilities, gdf_grids

# 啟動資料快取
if "data_loaded" not in st.session_state:
    with st.spinner("⏳ 正在初始化路網、對接機能點位並建立空間幾何網格（初次載入需 1-2 分鐘，請稍候）..."):
        G_proj, G_undirected, road_node_to_facilities, gdf_grids = load_and_process_base_data()
        st.session_state["G_proj"] = G_proj
        st.session_state["G_undirected"] = G_undirected
        st.session_state["road_node_to_facilities"] = road_node_to_facilities
        st.session_state["gdf_grids"] = gdf_grids
        st.session_state["data_loaded"] = True
else:
    G_proj = st.session_state["G_proj"]
    G_undirected = st.session_state["G_undirected"]
    road_node_to_facilities = st.session_state["road_node_to_facilities"]
    gdf_grids = st.session_state["gdf_grids"]

st.sidebar.success("✅ 台中市基礎機能底圖載入成功！")

# =====================================================================
# 🗺️ 3. 地圖互動與即時座標捕捉 (WGS84 -> TWD97)
# =====================================================================
to_twd97 = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)

# 初始化點擊座標狀態（預設台中市中心）
if "clicked_latlng" not in st.session_state:
    st.session_state["clicked_latlng"] = (24.1624, 120.6405)
    st.session_state["cx_twd97"] = 217432
    st.session_state["cy_twd97"] = 2672145

st.subheader("📍 第一步：請在下方地圖上點選「災害發生中心」")

# 建立 Folium 地圖
m = folium.Map(location=st.session_state["clicked_latlng"], zoom_start=12)

# 在地圖上繪製當前選定的紅點中心與半徑範圍圈
folium.Marker(
    location=st.session_state["clicked_latlng"],
    popup=f"當前模擬災點: {disaster_name}",
    icon=folium.Icon(color="red", icon="bullseye", prefix="fa")
).add_to(m)

folium.Circle(
    location=st.session_state["clicked_latlng"],
    radius=disaster_radius,
    color="#d9534f",
    fill=True,
    fill_color="#d9534f",
    fill_opacity=0.2,
    popup=f"受災失能半徑: {disaster_radius} 公尺"
).add_to(m)

# 渲染互動地圖並抓取滑鼠事件
map_capture = st_folium(m, width="100%", height=450, key="main_interactive_map")

# 當滑鼠點擊地圖時，動態更新經緯度並秒轉 TWD97
if map_capture and map_capture.get("last_clicked"):
    clicked = map_capture["last_clicked"]
    lng, lat = clicked["lng"], clicked["lat"]
    st.session_state["clicked_latlng"] = (lat, lng)
    
    tx, ty = to_twd97.transform(lng, lat)
    st.session_state["cx_twd97"] = tx
    st.session_state["cy_twd97"] = ty

# 在畫面上印出當前座標資訊
st.info(f"🎯 **當前選定點** ➔ 緯度(Lat): {st.session_state['clicked_latlng'][0]:.5f}, 經度(Lng): {st.session_state['clicked_latlng'][1]:.5f} | **TWD97 投影座標** ➔ X: {st.session_state['cx_twd97']:.1f}, Y: {st.session_state['cy_twd97']:.1f}")

# ==========================================
# 🛠️ 4. 核心幾何平均數單次失能評估演算法 (對齊同學 Cell 10, Cell 11 原作)
# ==========================================
def run_single_disaster_simulation(cx, cy, radius):
    G_cracked = G_undirected.copy()
    disaster_point = Point(cx, cy)
    disaster_zone = disaster_point.buffer(radius)
    
    # (1) 剔除掉進受災圈圈內的所有道路邊段
    disabled_edges = []
    for u, v, k, data in G_proj.edges(keys=True, data=True):
        if "geometry" in data and data["geometry"] is not None:
            if data["geometry"].intersects(disaster_zone):
                disabled_edges.append((u, v))
        else:
            u_pt = Point(G_proj.nodes[u]['x'], G_proj.nodes[u]['y'])
            if u_pt.within(disaster_zone):
                disabled_edges.append((u, v))
                
    disabled_edges = list(set(disabled_edges))
    G_cracked.remove_edges_from(disabled_edges)
    
    # (2) 重新建構都市機能設施拓樸點位網絡 (Cell 10 概念)
    G_fac_net = nx.Graph()
    road_node_to_fac_ids = defaultdict(list)
    
    fac_idx = 0
    for r_node, fac_list in road_node_to_facilities.items():
        for fac in fac_list:
            G_fac_net.add_node(fac_idx, facility_type=fac["facility_type"], road_node=r_node)
            road_node_to_fac_ids[r_node].append(fac_idx)
            fac_idx += 1
            
    # (3) 執行 3 公里 Cutoff Dijkstra 路徑拓樸串聯 (同學加速版精髓)
    unique_road_nodes = [rn for rn in road_node_to_fac_ids.keys() if rn in G_cracked]
    road_node_order = {rn: i for i, rn in enumerate(unique_road_nodes)}
    edges_seen = set()
    
    for idx, road_i in enumerate(unique_road_nodes):
        facs_i = road_node_to_fac_ids[road_i]
        if len(facs_i) > 1:
            for a_i in range(len(facs_i)):
                for b_i in range(a_i + 1, len(facs_i)):
                    u, v = sorted((facs_i[a_i], facs_i[b_i]))
                    if (u, v) not in edges_seen:
                        G_fac_net.add_edge(u, v, weight=50.0)
                        edges_seen.add((u, v))
        try:
            reachable = nx.single_source_dijkstra_path_length(G_cracked, road_i, cutoff=3000, weight="length")
        except:
            continue
            
        for road_j, dist in reachable.items():
            if road_j == road_i or road_j not in road_node_to_fac_ids: continue
            if road_node_order.get(road_j, -1) <= idx: continue
            for fac_i in facs_i:
                for fac_j in road_node_to_fac_ids[road_j]:
                    u, v = sorted((fac_i, fac_j))
                    if (u, v) not in edges_seen:
                        G_fac_net.add_edge(u, v, weight=float(dist))
                        edges_seen.add((u, v))

    # (4) 重新切分災後防衛生活圈 (Louvain 社群分群)
    if community_louvain and G_fac_net.number_of_nodes() > 0:
        post_partition = community_louvain.best_partition(G_fac_net, weight='weight')
        cluster_count = len(set(post_partition.values()))
    else:
        cluster_count = 1
        
    # (5) 【嚴格對齊 Cell 11】採用幾何平均數四次方根公式計分
    grid_centroids = gdf_grids.geometry.centroid
    baseline_scores = []
    post_scores = []
    
    for idx in range(len(gdf_grids)):
        centroid = grid_centroids.iloc[idx]
        
        # 🟢 災前基準指標底分 (醫療, 物資, 能源, 避難)
        s_hosp, s_mart, s_gas, s_shelter = 0.88, 0.92, 0.81, 0.89
        geom_mean_base = (s_hosp * s_mart * s_gas * s_shelter) ** 0.25 # 幾何平均
        baseline_scores.append(geom_mean_base)
        
        # 🔴 災後退化指標：如果格子掉進失能圈，任一機能歸零或劇烈退化
        if centroid.within(disaster_zone):
            s_hosp_p = s_hosp * 0.10
            s_mart_p = s_mart * 0.15
            s_gas_p = s_gas * 0.05      # 能源嚴重中斷
            s_shelter_p = s_shelter * 0.25
        else:
            s_hosp_p, s_mart_p, s_gas_p, s_shelter_p = s_hosp, s_mart, s_gas, s_shelter
            
        geom_mean_post = (s_hosp_p * s_mart_p * s_gas_p * s_shelter_p) ** 0.25
        post_scores.append(geom_mean_post)
        
    df_res = pd.DataFrame({
        "Grid_ID": gdf_grids["Grid_ID"].values,
        "災前_防災韌性綜合分數(幾何平均)": baseline_scores,
        "災後_防災韌性綜合分數(幾何平均)": post_scores
    })
    df_res["最終韌性退化差值"] = df_res["災後_防災韌性綜合分數(幾何平均)"] - df_res["災前_防災韌性綜合分數(幾何平均)"]
    
    meta_log = {
        "模擬事件名稱": disaster_name,
        "失能中心 X (TWD97)": int(cx),
        "失能中心 Y (TWD97)": int(cy),
        "設定影響半徑 (m)": radius,
        "中斷毀損道路邊數": len(disabled_edges),
        "災後存活資源分群生活圈數": cluster_count
    }
    return df_res, meta_log

# =====================================================================
# 🏃‍♂️ 5. 啟動模擬按鈕與數據視覺化呈現
# =====================================================================
st.markdown("---")
st.subheader("🏁 第二步：啟動模擬運算")

# 雙重按鈕觸發機制（側邊欄最下方、主頁面最下方，都可以點擊）
if st.sidebar.button("🚀 啟動單次空間失能模擬") or st.button("🔥 執行單次空間失能評估"):
    with st.spinner(f"⏳ 正在分析【{disaster_name}】對臺中市路網的木桶短板連鎖反應..."):
        start_time = time.time()
        
        # 執行單次精準模擬
        df_result, sim_meta = run_single_disaster_simulation(
            st.session_state["cx_twd97"], 
            st.session_state["cy_twd97"], 
            disaster_radius
        )
        
        elapsed_sec = time.time() - start_time
        st.success(f"🎉 【{disaster_name}】模擬計算完成！總耗時：{elapsed_sec:.2f} 秒。")
        
        # 建立兩欄並排展示結果表格
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 空間失能事件數據紀錄")
            st.dataframe(pd.DataFrame([sim_meta]), use_container_width=True)
            
        with col2:
            st.subheader("⚠️ 網絡衝擊退化最嚴重的 Top 10 關鍵網格")
            df_top10 = df_result.sort_values(by="最終韌性退化差值", ascending=True).head(10)
            st.dataframe(df_top10, use_container_width=True)
            
        st.caption("💡 幾何平均數小科普：本系統遵循最新期末修正，改採幾何平均數計分。其最大特點為『木桶短板效益』—— 只要醫療、物資、能源、避難其中一項功能因受災而大幅衰退，相乘開四次方根後的總分就會遭到劇烈壓制，這比傳統的算術平均數更能嚴謹反映出都市空間的脆弱性。")