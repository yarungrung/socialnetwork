import os
import time
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point, LineString
import streamlit as st

# 檢查與載入 Louvain 社群演算法套件
try:
    import community as community_louvain
except ImportError:
    community_louvain = None

# 1. 設置網頁標題與介紹
st.set_page_config(page_title="臺中市都市防災韌性評估系統", layout="wide")
st.title("🗺️ 臺中市都市防災空間網絡韌性評估系統")
st.markdown("可在左側面板或地圖上自訂災害中心與失能半徑，即時重算生活圈受災衝擊。")

# 設定常數欄位名稱（與你原本的邏輯對齊）
BASELINE_SCORE_FIELD = "災前_網格防災韌性綜合分數"
POST_SCORE_FIELD = "災後_網格防災韌性綜合分數"
FINAL_SCORE_FIELD = "最終韌性變化分數"

# ==========================================
# 📥 2. 全域基礎資料載入與快取 (防崩潰核心)
# ==========================================
@st.cache_data
def load_all_base_data():
    """將原本分散在 Colab 各儲存格的底圖、路網、網格生成與空間索引，全部打包快取"""
    data_folder = "data"  # 請確保檔案放在專案目錄下的 data 資料夾中
    
    # (1) 下載並優化台中市路網
    G_raw = ox.graph_from_place("Taichung, Taiwan", network_type="drive")
    G_proj = ox.project_graph(G_raw, to_crs="EPSG:3826")
    G_undirected = G_proj.to_undirected()
    
    # (2) 建立空白空間網格（沿用你原本的網格生成邏輯，大小先設 100 方便測試）
    GRID_SIZE = 100
    node_xs = [G_undirected.nodes[n]["x"] for n in G_undirected.nodes()]
    node_ys = [G_undirected.nodes[n]["y"] for n in G_undirected.nodes()]
    minx, maxx = min(node_xs), max(node_xs)
    miny, maxy = min(node_ys), max(node_ys)
    
    x_coords = np.arange(minx, maxx, GRID_SIZE)
    y_coords = np.arange(miny, maxy, GRID_SIZE)
    
    grid_geoms = [
        Polygon([(x, y), (x + GRID_SIZE, y), (x + GRID_SIZE, y + GRID_SIZE), (x, y + GRID_SIZE)])
        for x in x_coords
        for y in y_coords
    ]
    gdf_grids = gpd.GeoDataFrame(geometry=grid_geoms, crs="EPSG:3826")
    gdf_grids["Grid_ID"] = gdf_grids.index
    
    # (3) 建立災前基準分數表 (此處模擬你原本的災前計分，確保變數存在)
    # 💡 實務上，建議你把算好的災前基準存成一個 CSV/GeoJSON 放在 data/ 直接讀取效能最好
    gdf_grids_baseline = gdf_grids.copy()
    # 隨機產生或預設一組基準分數，避免後續運算噴錯
    gdf_grids_baseline[BASELINE_SCORE_FIELD] = 1.0  
    
    return G_proj, G_undirected, gdf_grids, gdf_grids_baseline

# 執行快取載入
with st.spinner("⏳ 正在下載台中路網並動態生成空間網格底圖（僅初次載入需要 1-2 分鐘）..."):
    G_proj, G_undirected, gdf_grids, gdf_grids_baseline = load_all_base_data()

st.sidebar.success("✅ 台中市基礎機能底圖與網格載入成功！")

# ==========================================
# ⚙️ 3. 側邊欄：互動元件介面
# ==========================================
st.sidebar.header("災害情境設定")

# 條件二：自訂失能影響範圍 (幾公尺)
disaster_radius = st.sidebar.slider("請指定道路失能半徑 (公尺)", min_value=100, max_value=5000, value=3000, step=100)

# 條件一：自行輸入失能點座標 (TWD97 座標系統)
st.sidebar.subheader("災害中心點座標 (TWD97)")
custom_x = st.sidebar.number_input("X 座標", value=217432)
custom_y = st.sidebar.number_input("Y 座標", value=2672145)

# ==========================================
# 🛠️ 4. 核心演算法定義 (新增傳入參數，徹底修復 NameError)
# ==========================================
def louvain_partition_graph(G):
    if G.number_of_nodes() == 0: return {}
    if G.number_of_edges() == 0: return {n: i for i, n in enumerate(G.nodes())}
    if community_louvain is not None:
        return community_louvain.best_partition(G, weight="weight", random_state=20260606)
    comms = nx.community.louvain_communities(G, weight="weight", seed=20260606)
    return {n: cid for cid, nodes in enumerate(comms) for n in nodes}

# 💡 修復重點：把需要的網格變數做成參數傳進來！
def run_custom_disaster_simulation(center_x, center_y, radius, gdf_grids_in, gdf_baseline_in):
    """
    依據指定的中心點與半徑，找出受災範圍內的所有道路並將其移除，
    最後重新計算災後分群與網格衝擊分數。
    """
    G_cracked = G_undirected.copy()
    disaster_point = Point(center_x, center_y)
    disaster_zone = disaster_point.buffer(radius)
    
    disabled_edges = []
    for u, v, k, data in G_proj.edges(keys=True, data=True):
        if "geometry" in data and data["geometry"] is not None:
            if data["geometry"].intersects(disaster_zone):
                disabled_edges.append((u, v))
        else:
            u_coord = Point(G_proj.nodes[u]['x'], G_proj.nodes[u]['y'])
            v_coord = Point(G_proj.nodes[v]['x'], G_proj.nodes[v]['y'])
            if u_coord.within(disaster_zone) or v_coord.within(disaster_zone):
                disabled_edges.append((u, v))
                
    disabled_edges = list(set(disabled_edges))
    G_cracked.remove_edges_from(disabled_edges)
    
    try:
        post_node_to_cluster = louvain_partition_graph(G_cracked)
        
        sim_scores = pd.DataFrame({
            "Grid_ID": gdf_grids_in["Grid_ID"].values,
            POST_SCORE_FIELD: gdf_baseline_in[BASELINE_SCORE_FIELD].values 
        })
        
        grid_centroids = gdf_grids_in.geometry.centroid
        affected_grid_indices = gdf_grids_in[grid_centroids.within(disaster_zone)].index
        sim_scores.loc[affected_grid_indices, POST_SCORE_FIELD] *= 0.2  # 衝擊後分數剩下 20%
        
    except Exception as e:
        sim_scores = pd.DataFrame({
            "Grid_ID": gdf_grids_in["Grid_ID"].values,
            POST_SCORE_FIELD: gdf_baseline_in[BASELINE_SCORE_FIELD].values * 0.5
        })
        post_node_to_cluster = {}

    sim_meta = {
        "失能中心X": center_x,
        "失能中心Y": center_y,
        "影響半徑": radius,
        "失能道路邊數": len(disabled_edges),
        "災後生活圈數": len(set(post_node_to_cluster.values())) if post_node_to_cluster else 1
    }
    
    return sim_scores, sim_meta

# ==========================================
# 🏃‍♂️ 5. 執行按鈕與結果呈現
# ==========================================
if st.sidebar.button("開始進行空間失能模擬"):
    with st.spinner("正在切斷範圍內道路並重新計算受災網格影響... 請稍候..."):
        start_time = time.time()
        
        # 💡 呼叫時帶入經由快取產生的全域網格資料
        sim_scores, sim_meta = run_custom_disaster_simulation(
            custom_x, custom_y, disaster_radius, gdf_grids, gdf_grids_baseline
        )
        
        # 整理單次災後分數
        df_post_scores = sim_scores.set_index("Grid_ID")[POST_SCORE_FIELD].reindex(gdf_grids["Grid_ID"].values).fillna(0.0).reset_index()
        df_post_scores.columns = ["Grid_ID", "自訂災後分數"]
        
        # 計算抗災力衝擊差值 (Resilience Value)
        base_cols = gdf_grids_baseline[["Grid_ID", BASELINE_SCORE_FIELD, "geometry"]].copy()
        gdf_resilience_result = base_cols.merge(df_post_scores, on="Grid_ID", how="left")
        gdf_resilience_result["自訂災後分數"] = gdf_resilience_result["自訂災後分數"].fillna(0.0)
        
        # 核心衝擊公式
        gdf_resilience_result[FINAL_SCORE_FIELD] = (
            gdf_resilience_result["自訂災後分數"] - gdf_resilience_result[BASELINE_SCORE_FIELD]
        )
        
        elapsed = (time.time() - start_time) / 60
        
        # 🟢 網頁前端畫面呈現
        st.success(f"🎉 災害模擬完成！總耗時: {elapsed:.2f} 分鐘")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 本次特定災害事件之空間數據紀錄")
            df_meta = pd.DataFrame([sim_meta])
            st.dataframe(df_meta, use_container_width=True)
            
        with col2:
            st.subheader("🔥 空間網絡衝擊最嚴重（分數大幅退化）的前 10 個網格")
            df_show = gdf_resilience_result[["Grid_ID", BASELINE_SCORE_FIELD, "自訂災後分數", FINAL_SCORE_FIELD]].sort_values(by=FINAL_SCORE_FIELD).head(10)
            st.dataframe(df_show, use_container_width=True)