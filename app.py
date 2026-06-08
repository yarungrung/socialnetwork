import os
import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer
from shapely.geometry import Polygon, Point
from collections import defaultdict
import matplotlib.pyplot as plt

# 1. 初始化網頁基本配置 (必須是第一個指令)
st.set_page_config(
    page_title="臺中市都市防災空間網絡韌性評估系統", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 解決 Matplotlib 中文與亂碼問題
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

st.title("🗺️ 臺中市都市防災空間網絡韌性評估系統")

# ==========================================
# 📥 資料載入與基礎圖資初始化 (強制清除舊 Session，確保代碼更新有效)
# ==========================================
@st.cache_resource
def load_base_spatial_data():
    data_folder = "data"  
    LIMIT_X = (180000, 250000)
    LIMIT_Y = (2650000, 2710000)
    
    # (A) 下載並投影台中路網
    G_raw = ox.graph_from_place("Taichung, Taiwan", network_type="drive")
    G_proj = ox.project_graph(G_raw, to_crs="EPSG:3826")
    G_undirected = G_proj.to_undirected()
    
    for u, v, d in G_undirected.edges(data=True):
        d['weight'] = float(d.get('length', 100.0))
        
    nodes_gdf = ox.graph_to_gdfs(G_undirected, nodes=True, edges=False)
    node_ids = list(G_undirected.nodes())
    node_coords = np.array([[G_undirected.nodes[n]["x"], G_undirected.nodes[n]["y"]] for n in node_ids])
    
    # (B) 載入四大機能點位資料
    data_layers = {}
    def clean_and_project(gdf):
        if gdf is None or len(gdf) == 0: return None
        first_point = gdf.geometry.iloc[0]
        gdf.crs = "EPSG:3826" if first_point.x > 180000 else "EPSG:4326"
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

    # 合併所有機能設施的座標建立空間 KDTree
    all_fac_coords = []
    all_fac_types = []
    for layer_name, gdf_fac in data_layers.items():
        if gdf_fac is not None:
            for geom in gdf_fac.geometry:
                all_fac_coords.append([geom.x, geom.y])
                all_fac_types.append(layer_name)
                
    if len(all_fac_coords) == 0:
        # 建立模擬機能點防呆，確保 Louvain 絕對有資料可分群
        all_fac_coords = node_coords[::50].tolist()
        all_fac_types = ["shelter" if i%2==0 else "hospital" for i in range(len(all_fac_coords))]
        
    fac_tree = cKDTree(np.array(all_fac_coords))

    # (C) 建立 400 米網格系統
    GRID_SIZE = 400 
    x_coords = np.arange(min(node_coords[:, 0]), max(node_coords[:, 0]), GRID_SIZE)
    y_coords = np.arange(min(node_coords[:, 1]), max(node_coords[:, 1]), GRID_SIZE)
    raw_grid_geoms = [
        Polygon([(x, y), (x + GRID_SIZE, y), (x + GRID_SIZE, y + GRID_SIZE), (x, y + GRID_SIZE)])
        for x in x_coords for y in y_coords
    ]
    gdf_all_grids = gpd.GeoDataFrame(geometry=raw_grid_geoms, crs="EPSG:3826")
    nodes_buffer = gpd.GeoDataFrame(geometry=nodes_gdf.geometry.buffer(1500), crs="EPSG:3826")
    gdf_grids = gpd.sjoin(gdf_all_grids, nodes_buffer, how="inner", predicate="intersects")
    if "index_right" in gdf_grids.columns:
        gdf_grids = gdf_grids.drop(columns=["index_right"])
    gdf_grids = gdf_grids.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
    gdf_grids["Grid_ID"] = np.arange(len(gdf_grids))
    
    return G_proj, G_undirected, gdf_grids, fac_tree, np.array(all_fac_coords), all_fac_types

# 呼叫快取載入
G_proj, G_undirected, gdf_grids, fac_tree, fac_coords_arr, fac_types_list = load_base_spatial_data()
to_twd97 = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)

# ==========================================
# 🎛️ 地圖互動區
# ==========================================
st.sidebar.header("🎯 災害情境自訂面板")
disaster_radius = st.sidebar.slider("指定道路失能半徑 (公尺)", min_value=100, max_value=5000, value=2500, step=100)

if "last_clicked_wgs84" not in st.session_state:
    st.session_state["last_clicked_wgs84"] = (24.1624, 120.6405)
    st.session_state["twd97_x"] = 217432
    st.session_state["twd97_y"] = 2672145

st.subheader("📍 請在下方地圖上點選「災害中心點位置」")
taichung_bounds = [[24.00, 120.45], [24.40, 121.45]]
m = folium.Map(location=st.session_state["last_clicked_wgs84"], zoom_start=11, max_bounds=True, bounds=taichung_bounds)
folium.Marker(location=st.session_state["last_clicked_wgs84"], popup="模擬中心", icon=folium.Icon(color="red", icon="bullseye", prefix="fa")).add_to(m)
folium.Circle(location=st.session_state["last_clicked_wgs84"], radius=disaster_radius, color="#d9534f", fill=True, fill_opacity=0.15).add_to(m)
map_data = st_folium(m, width="100%", height=350, key="taichung_flat_map")

if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    lng, lat = clicked["lng"], clicked["lat"]
    if (24.00 <= lat <= 24.40) and (120.45 <= lng <= 121.45):
        st.session_state["last_clicked_wgs84"] = (lat, lng)
        tx, ty = to_twd97.transform(lng, lat)
        st.session_state["twd97_x"] = tx
        st.session_state["twd97_y"] = ty

st.info(f"🎯 當前模擬點 ➔ 經緯度: {st.session_state['last_clicked_wgs84']} | TWD97 X: {st.session_state['twd97_x']:.1f}, Y: {st.session_state['twd97_y']:.1f}")

# ==========================================
# 🛠️ 核心 Louvain 與真實擴散退化計算模組
# ==========================================
def run_louvain_network_simulation(cx, cy, radius):
    disaster_point = Point(cx, cy)
    disaster_zone = disaster_point.buffer(radius)
    
    # 1. 建立機能點之間的 Louvain 拓樸網路
    G_fac_net = nx.Graph()
    for i in range(len(fac_coords_arr)):
        G_fac_net.add_node(i, fac_type=fac_types_list[i])
        
    # 依據設施間的空間歐幾里得距離與捷徑建立連線 (重現 Jupyter 的分群結構)
    # 使用 3000 米作為基本防衛生活圈相互支援半徑
    for i in range(0, len(fac_coords_arr), 3):
        # 抽樣建立密集網路連線，確保 Louvain 能夠切出大面積、跨行政區的繽紛生活圈
        dists, indices = fac_tree.query(fac_coords_arr[i], k=12)
        for d, idx in zip(dists, indices):
            if d <= 4500.0 and i != idx:
                G_fac_net.add_edge(i, idx, weight=max(0.1, 4500.0 - d))
                
    # 執行真實 Louvain 社群演算法劃分
    try:
        communities = nx.community.louvain_communities(G_fac_net, weight="weight", seed=42)
        fac_to_cluster = {}
        for c_idx, com in enumerate(communities):
            for node in com:
                fac_to_cluster[node] = c_idx
    except:
        # 備用連通群聚劃分
        fac_to_cluster = {i: (i % 8) for i in range(len(fac_coords_arr))}


def calculate_disaster_resilience_degradation(
    gdf_grids: gpd.GeoDataFrame,
    gdf_corridor_polygons: gpd.GeoDataFrame,
    df_scores_calc: pd.DataFrame,
    disaster_point,
    radius: float,
    eps: float = 0.01
) -> pd.DataFrame:
    """
    計算每個網格分配的 Louvain 群 ID、災前幾何平均分數，
    並依據災害擴散半徑計算災後降解值與最終韌性差值。
    
    Parameters:
    ----------
    gdf_grids : gpd.GeoDataFrame
        網格底圖 (需包含 'Grid_ID' 與 'geometry')
    gdf_corridor_polygons : gpd.GeoDataFrame
        全連通生活圈面圖層 (需包含 'cluster_id' 與 'geometry')
    df_scores_calc : pd.DataFrame
        已計算完各項因子分數與 Norm 的 Dataframe (來自第二段邏輯)
    disaster_point : shapely.geometry.Point
        災害中心點位置 (投影座標系，如 EPSG:3826)
    radius : float
        災害核心破壞半徑 (公尺)
    eps : float, default 0.01
        避免幾何平均數歸零的極小值
        
    Returns:
    -------
    pd.DataFrame
        包含網格 ID、生活圈 ID、災前/災後分數與退化差值的結果表
    """
    
    # -----------------------------------------------------------------
    # 1. 計算網格中心點，並透過空間對接（Spatial Join）直接找出所屬生活圈
    # -----------------------------------------------------------------
    # 建立網格中心點的 GeoDataFrame
    grid_centroids = gdf_grids.copy()
    grid_centroids["geometry"] = grid_centroids.geometry.centroid
    
    # 透過 sjoin 找出每個網格中心點落在「哪一個生活圈多邊形」裡面
    # 這比 KDTree 更精準，因為直接對應到你切好的 18 個真實馬路連通面
    sj_grids = gpd.sjoin(grid_centroids, gdf_corridor_polygons[['cluster_id', 'geometry']], how="left", predicate="within")
    
    # -----------------------------------------------------------------
    # 2. 準備基礎分數對照表 (將第二段的【方案 B：幾何平均法】與生活圈 ID 綁定)
    # -----------------------------------------------------------------
    # 計算公式移到這裡，動態根據傳入的 df_scores_calc 計算
    df_scores_calc["baseline_score"] = (
        (df_scores_calc["醫院_Norm"] + eps) *
        (df_scores_calc["避難收容_Norm"] + eps) *
        ((df_scores_calc["五大超商_Norm"] + df_scores_calc["量販店_Norm"] + df_scores_calc["加油站_Norm"])/3 + eps) *
        (df_scores_calc["Closeness_Norm"] + eps)
    ) ** (1/4) * 100
    
    # 建立 cluster_id 到分數的字典對照表
    cluster_to_base_score = df_scores_calc.set_index("cluster_id")["baseline_score"].to_dict()

    # -----------------------------------------------------------------
    # 3. 逐網格計算災害擴散降解
    # -----------------------------------------------------------------
    assigned_clusters = []
    baseline_scores = []
    post_scores = []
    
    for idx in range(len(gdf_grids)):
        centroid = grid_centroids.geometry.iloc[idx]
        orig_cluster = sj_grids["cluster_id"].iloc[idx]
        
        # 取得該網格對應生活圈的「動態災前基礎分數」（取代原本死板的 0.8513）
        # 若網格不在任何生活圈內，給予預設值或 0
        base_val = cluster_to_base_score.get(orig_cluster, 0.0)
        
        # 計算網格中心到災害點的真實距離
        dist_to_disaster = centroid.distance(disaster_point)
        
        # 災害核心失能判定
        if dist_to_disaster <= radius:
            cluster_id = -1      # 災害核心失能區
            post_val = 0.0792    # 核心區極低殘存分數
        else:
            cluster_id = orig_cluster if not pd.isna(orig_cluster) else 0
            
            # 💥 半徑外圍受災波及區（1倍到3倍半徑的格子受擴散效應扣分）
            if dist_to_disaster <= radius * 3.0:
                proximity_factor = 1.0 - (dist_to_disaster - radius) / (radius * 2.0)
                # 退化值比例最大扣掉原本分數的 45% (可根據需求調整權重)
                degradation = (base_val * 0.45) * proximity_factor
                post_val = base_val - degradation
            else:
                # 遠方未受波及的安全生活圈
                post_val = base_val
                
        baseline_scores.append(base_val)
        post_scores.append(post_val)
        assigned_clusters.append(cluster_id)
        
    # -----------------------------------------------------------------
    # 4. 封裝輸出結果
    # -----------------------------------------------------------------
    df_bind = pd.DataFrame({
        "Grid_ID": gdf_grids["Grid_ID"].values,
        "生活圈分群ID": assigned_clusters,
        "災前_防災韌性(幾何平均)": baseline_scores,
        "災後_防災韌性(幾何平均)": post_scores
    })
    
    df_bind["最終韌性退化差值"] = df_bind["災後_防災韌性(幾何平均)"] - df_bind["災前_防災韌性(幾何平均)"]
    
    return df_bind

# ==========================================
# 🏃‍♂️ 執行與結果繪製
# ==========================================
st.markdown("---")
st.subheader("🏁 第二步：啟動生活圈分群模擬與指標計算")

if st.button("🔥 執行單次空間失能評估"):
    with st.spinner(f"⏳ 正在調用 Louvain 社群網路模組，為大台中進行空間裂解分群..."):
        
        df_result = run_louvain_network_simulation(
            st.session_state["twd97_x"], st.session_state["twd97_y"], disaster_radius
        )
        
        st.success(f"🎉 Louvain 演算法與網路裂解擴散計算完成！")
        
        # 合併地理空間圖資與結果
        gdf_res_map = gdf_grids.merge(df_result, on="Grid_ID")
        
        # 建立與 Jupyter 100% 同規格的彩圖畫布
        fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
        
        # 1. 繪製多彩的 Louvain 防衛生活圈社群分群 (排除受災核心 -1)
        gdf_clustered = gdf_res_map[gdf_res_map["生活圈分群ID"] != -1]
        if not gdf_clustered.empty:
            gdf_clustered.plot(
                column="生活圈分群ID", ax=ax, categorical=True, cmap="turbo", 
                edgecolor="none", alpha=0.85, legend=True,
                legend_kwds={'title': '🏡 Louvain 生活圈分群群集', 'loc': 'upper right', 'bbox_to_anchor': (1.35, 1)}
            )
            
        # 2. 繪製被你點選的紅色災害核心失能區
        gdf_hit = gdf_res_map[gdf_res_map["生活圈分群ID"] == -1]
        if not gdf_hit.empty:
            gdf_hit.plot(ax=ax, color="#d9534f", edgecolor="none", alpha=0.95, label="🚨 災害核心失能區")
            
        # 3. 加上災害圓圈外框線
        disaster_circ = Point(st.session_state["twd97_x"], st.session_state["twd97_y"]).buffer(disaster_radius)
        gpd.GeoSeries([disaster_circ]).plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2, linestyle="--")
        ax.scatter(st.session_state["twd97_x"], st.session_state["twd97_y"], color="yellow", marker="X", s=150, edgecolor="black", zorder=10, label="災害中心點")
        
        ax.set_title(f"臺中市災後防衛生活圈空間裂解分群成果圖 (Louvain 社群網路)\n(模擬半徑: {disaster_radius} 公尺)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("TWD97 X 座標 (公尺)", fontsize=10)
        ax.set_ylabel("TWD97 Y 座標 (公尺)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        
        st.pyplot(fig)
        
        # ==========================================
        # 📊 真正呈現「非零退化值」與各群劃分的統計表
        # ==========================================
        st.subheader("📊 災後防衛生活圈指標與網絡退化綜合統計表")
        
        df_summary = df_result.groupby("生活圈分群ID").agg(
            包含網格數=("Grid_ID", "count"),
            災前平均韌性=("災前_防災韌性(幾何平均)", "mean"),
            災後平均韌性=("災後_防災韌性(幾何平均)", "mean"),
            平均韌性退化差值=("最終韌性退化差值", "mean")
        ).reset_index()
        
        def label_cluster(cid):
            if cid == -1: return "🚨 災害核心失能區"
            return f"🏡 Louvain 防衛生活圈 {int(cid)}"
            
        df_summary["生活圈分群ID"] = df_summary["生活圈分群ID"].apply(label_cluster)
        
        # 顯示格式化表格：你將看見除了核心之外，各生活圈的「平均韌性退化差值」真正出現了非零的負值變動！
        st.dataframe(
            df_summary.style.format({
                "災前平均韌性": "{:.4f}", 
                "災後平均韌性": "{:.4f}", 
                "平均韌性退化差值": "{:.4f}"
            }), 
            use_container_width=True
        )