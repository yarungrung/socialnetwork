import os
import streamlit as st
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

# 1. 初始化網頁基本配置 (必須是 Streamlit 的第一個指令)
st.set_page_config(
    page_title="臺中市都市防災空間網絡韌性評估系統", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 建立資料快取功能 (從真實圖資中加載)
@st.cache_data(show_spinner=False)
def load_and_process_base_data():
    data_folder = "data" # 對齊 GitHub 相對路徑
    LIMIT_X = (180000, 250000)
    LIMIT_Y = (2650000, 2710000)
    
    # (A) 下載與投影台中路網
    G_raw = ox.graph_from_place("Taichung, Taiwan", network_type="drive")
    G_proj = ox.project_graph(G_raw, to_crs="EPSG:3826")
    G_undirected = G_proj.to_undirected()
    
    # (B) 讀取與過濾機能點位
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

    # 避難所
    shelter_path = os.path.join(data_folder, "臺中市避難收容所位置及收容人數_CSV.csv")
    if os.path.exists(shelter_path):
        df = pd.read_csv(shelter_path, encoding="utf-8-sig")
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.iloc[:, 0], df.iloc[:, 1]))
        data_layers["shelter"] = clean_and_project(gdf)
        
    # 醫院
    hosp_path = os.path.join(data_folder, "醫院.shp")
    if os.path.exists(hosp_path):
        data_layers["hospital"] = clean_and_project(gpd.read_file(hosp_path, encoding="cp950"))
        
    # 量販店
    mart_path = os.path.join(data_folder, "台中量販店.shp")
    if os.path.exists(mart_path):
        data_layers["mart"] = clean_and_project(gpd.read_file(mart_path, encoding="cp950"))
        
    # 加油站
    gas_path = os.path.join(data_folder, "加油站.csv")
    if os.path.exists(gas_path):
        df = pd.read_csv(gas_path, encoding="utf-8-sig")
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.iloc[:, 0], df.iloc[:, 1]))
        data_layers["gas"] = clean_and_project(gdf)

    # (C) 建立 cKDTree 空間對接路網
    node_ids = list(G_undirected.nodes())
    node_coords = np.array([[G_undirected.nodes[n]["x"], G_undirected.nodes[n]["y"]] for n in node_ids])
    road_tree = cKDTree(node_coords)
    
    road_node_to_facilities = {}
    for layer_name, gdf_fac in data_layers.items():
        if gdf_fac is None: continue
        fac_coords = np.array([[geom.x, geom.y] for geom in gdf_fac.geometry])
        distances, indices = road_tree.query(fac_coords)
        for idx, (dist, node_idx) in enumerate(zip(distances, indices)):
            if dist <= 100.0:
                target_node = node_ids[node_idx]
                if target_node not in road_node_to_facilities:
                    road_node_to_facilities[target_node] = []
                road_node_to_facilities[target_node].append({
                    "facility_type": layer_name,
                    "data_row": gdf_fac.iloc[idx].to_dict()
                })

    # (D) 生成均勻空間網格 (展示先用 200，交報告可改 100)
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

# 執行快取並存入共用記憶體
if "initialized" not in st.session_state:
    with st.spinner("⏳ 正在下載臺中市路網、對接機能點位（初次載入需 1-2 分鐘）..."):
        G_proj, G_undirected, road_node_to_facilities, gdf_grids = load_and_process_base_data()
        st.session_state["G_proj"] = G_proj
        st.session_state["G_undirected"] = G_undirected
        st.session_state["road_node_to_facilities"] = road_node_to_facilities
        st.session_state["gdf_grids"] = gdf_grids
        st.session_state["initialized"] = True

# 3. 🗺️ 關鍵核心：在側邊欄最上方製作自訂選單切換
st.sidebar.title(" NAVIGATION 導覽選單")
page = st.sidebar.radio("請選擇要檢視的頁面：", ["🏠 系統首頁說明", "📊 災害模擬主程式"])

# 依據選單內容，直接把對應的程式碼渲染出來
if page == "🏠 系統首頁說明":
    st.title("🗺️ 臺中市都市防災空間網絡韌性評估系統")
    st.markdown("""
    本系統整合了 **OSM 道路拓樸網絡**、**都市防災害機能點位空間對接** 以及 **Louvain 社群分群演算法**。
    
    ### 💡 快速開始指南：
    1. 請點選左側邊欄選單切換至 **[📊 災害模擬主程式]**。
    2. 切換後，左側將會跳出完整的「半徑調整拉桿」與「執行按鈕」。
    3. 您可以直接在右側出現的臺中市地圖上任意點選，即可立刻進行單次幾何平均數防災模擬！
    """)
    st.success("✅ 系統環境與真實圖資已成功就位！請由左側切換至模擬主程式。")

elif page == "📊 災害模擬主程式":
    # 直接在 app.py 裡呼叫同層級的 mainpage.py 內容
    import mainpage
    mainpage.show_simulation_page()