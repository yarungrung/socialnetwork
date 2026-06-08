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
import matplotlib.cm as cm

# 1. 初始化網頁基本配置 (必須是第一個指令)
st.set_page_config(
    page_title="臺中市都市防災空間網絡韌性評估系統", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 解決 Matplotlib 中文顯示問題
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

st.title("🗺️ 臺中市都市防災空間網絡韌性評估系統")

# 固定災害名稱變數
disaster_name = "突發性空間失能事件"

# ==========================================
# 📥 資料載入與初始化區 (整合至 Session State)
# ==========================================
if "initialized" not in st.session_state:
    with st.spinner("⏳ 正在加載臺中市路網與機能圖資（首次啟動約需 1 分鐘，請稍候）..."):
        try:
            data_folder = "data"  
            LIMIT_X = (180000, 250000)
            LIMIT_Y = (2650000, 2710000)
            
            # (A) 下載與投影台中路網
            G_raw = ox.graph_from_place("Taichung, Taiwan", network_type="drive")
            G_proj = ox.project_graph(G_raw, to_crs="EPSG:3826")
            G_undirected = G_proj.to_undirected()
            
            # 給每條邊一個預設的 weight 欄位，確保演算法能跑
            for u, v, d in G_undirected.edges(data=True):
                d['weight'] = float(d.get('length', 100.0))
            
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
                    if dist <= 150.0: # 稍微放寬容許度，讓點位能更好地黏上路網
                        target_node = node_ids[node_idx]
                        if target_node not in road_node_to_facilities:
                            road_node_to_facilities[target_node] = []
                        road_node_to_facilities[target_node].append({
                            "facility_type": layer_name,
                            "data_row": gdf_fac.iloc[idx].to_dict()
                        })

            # (D) 動態生成空間網格 (200m)
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
            
            # 將成功載入的物件寫入 Session State 中
            st.session_state["G_proj"] = G_proj
            st.session_state["G_undirected"] = G_undirected
            st.session_state["road_node_to_facilities"] = road_node_to_facilities
            st.session_state["gdf_grids"] = gdf_grids
            st.session_state["initialized"] = True
            
        except Exception as e:
            st.error(f"❌ 圖資下載或初始化失敗，錯誤訊息: {str(e)}")
            st.stop()

# 提取初始化後的資料
G_proj = st.session_state["G_proj"]
G_undirected = st.session_state["G_undirected"]
road_node_to_facilities = st.session_state["road_node_to_facilities"]
gdf_grids = st.session_state["gdf_grids"]

# 設定座標轉換器: WGS84 -> TWD97 (EPSG:3826)
to_twd97 = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)

# ==========================================
# 🎛️ 側邊欄控制面板
# ==========================================
st.sidebar.header("🎯 災害情境自訂面板")
disaster_radius = st.sidebar.slider("指定道路失能半徑 (公尺)", min_value=100, max_value=5000, value=2000, step=100)

# 初始化點擊座標
if "last_clicked_wgs84" not in st.session_state:
    st.session_state["last_clicked_wgs84"] = (24.1624, 120.6405)
    st.session_state["twd97_x"] = 217432
    st.session_state["twd97_y"] = 2672145

# ==========================================
# 🗺️ 主頁面互動點選地圖
# ==========================================
st.subheader("📍 請在下方地圖上點選「災害中心點位置」")

taichung_bounds = [[24.00, 120.45], [24.40, 121.45]]

m = folium.Map(
    location=st.session_state["last_clicked_wgs84"], 
    zoom_start=11,
    min_zoom=10,
    max_bounds=True,
    max_bounds_viscosity=1.0,
    bounds=taichung_bounds
)

folium.Marker(
    location=st.session_state["last_clicked_wgs84"],
    popup=f"模擬中心",
    icon=folium.Icon(color="red", icon="bullseye", prefix="fa")
).add_to(m)

folium.Circle(
    location=st.session_state["last_clicked_wgs84"],
    radius=disaster_radius,
    color="#d9534f",
    fill=True,
    fill_color="#d9534f",
    fill_opacity=0.2,
).add_to(m)

map_data = st_folium(m, width="100%", height=350, key="taichung_flat_map")

if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    lng, lat = clicked["lng"], clicked["lat"]
    
    if (24.00 <= lat <= 24.40) and (120.45 <= lng <= 121.45):
        st.session_state["last_clicked_wgs84"] = (lat, lng)
        tx, ty = to_twd97.transform(lng, lat)
        st.session_state["twd97_x"] = tx
        st.session_state["twd97_y"] = ty

st.info(f"🎯 **當前選定點** ➔ 緯度(Lat): {st.session_state['last_clicked_wgs84'][0]:.5f}, 經度(Lng): {st.session_state['last_clicked_wgs84'][1]:.5f} | **TWD97 投影座標** ➔ X: {st.session_state['twd97_x']:.1f}, Y: {st.session_state['twd97_y']:.1f}")

# ==========================================
# 🛠️ 核心運算演算法 (重構防衛生活圈分裂邏輯)
# ==========================================
def run_single_disaster_simulation(cx, cy, radius):
    G_cracked = G_undirected.copy()
    disaster_point = Point(cx, cy)
    disaster_zone = disaster_point.buffer(radius)
    
    # 1. 識別並徹底移除受災邊
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
    
    # 2. 建立生活圈設施節點
    G_fac_net = nx.Graph()
    road_node_to_fac_ids = defaultdict(list)
    
    fac_idx = 0
    for r_node, fac_list in road_node_to_facilities.items():
        if r_node not in G_cracked: continue
        for fac in fac_list:
            G_fac_net.add_node(fac_idx, facility_type=fac["facility_type"], road_node=r_node)
            road_node_to_fac_ids[r_node].append(fac_idx)
            fac_idx += 1
            
    # 3. 限制 Dijkstra 對接距離為 1500 米（超出即代表無法互相支援，形成生活圈分裂裂解）
    unique_road_nodes = list(road_node_to_fac_ids.keys())
    edges_seen = set()
    
    for idx, road_i in enumerate(unique_road_nodes):
        facs_i = road_node_to_fac_ids[road_i]
        
        # 同一節點上的設施直接強連結
        if len(facs_i) > 1:
            for a_i in range(len(facs_i)):
                for b_i in range(a_i + 1, len(facs_i)):
                    u, v = sorted((facs_i[a_i], facs_i[b_i]))
                    G_fac_net.add_edge(u, v, weight=1.0)
        
        # 使用更嚴格的 1500m 生活圈截斷距離，逼出多個分群
        try:
            reachable = nx.single_source_dijkstra_path_length(G_cracked, road_i, cutoff=1500, weight="weight")
        except:
            continue
            
        for road_j, dist in reachable.items():
            if road_j == road_i or road_j not in road_node_to_fac_ids: continue
            for fac_i in facs_i:
                for fac_j in road_node_to_fac_ids[road_j]:
                    u, v = sorted((fac_i, fac_j))
                    if (u, v) not in edges_seen:
                        # 距離越短，權重（親密度）越高
                        inv_weight = max(0.001, 1500.0 - dist)
                        G_fac_net.add_edge(u, v, weight=inv_weight)
                        edges_seen.add((u, v))

    # 4. 採用正統社群偵測，若套件不支援則用連通分支但加上距離限制
    post_partition = {}
    try:
        import community as community_louvain
        if G_fac_net.number_of_nodes() > 0:
            post_partition = community_louvain.best_partition(G_fac_net, weight='weight')
    except:
        if G_fac_net.number_of_nodes() > 0:
            components = list(nx.connected_components(G_fac_net))
            for comp_idx, comp in enumerate(components):
                for fac_node in comp:
                    post_partition[fac_node] = comp_idx
    
    # 5. 將網格歸類到所屬的分群
    grid_centroids = gdf_grids.geometry.centroid
    node_ids = list(G_undirected.nodes())
    node_coords = np.array([[G_undirected.nodes[n]["x"], G_undirected.nodes[n]["y"]] for n in node_ids])
    road_tree = cKDTree(node_coords)
    
    baseline_scores = []
    post_scores = []
    assigned_clusters = []
    
    for idx in range(len(gdf_grids)):
        centroid = grid_centroids.iloc[idx]
        
        s1_b, s2_b, s3_b, s4_b = 0.85, 0.90, 0.78, 0.88
        geom_mean_base = (s1_b * s2_b * s3_b * s4_b) ** 0.25
        baseline_scores.append(geom_mean_base)
        
        if centroid.within(disaster_zone):
            s1_p, s2_p, s3_p, s4_p = s1_b * 0.12, s2_b * 0.20, s3_b * 0.05, s4_b * 0.30
            cluster_id = -1  # 災區中心直接劃為受災失能孤立區
        else:
            s1_p, s2_p, s3_p, s4_p = s1_b, s2_b, s3_b, s4_b
            
            # 尋找最近路網節點的生活圈分群
            dist, node_idx = road_tree.query([centroid.x, centroid.y])
            nearest_road_node = node_ids[node_idx]
            
            cluster_id = -1
            # 只有在距離小於 1500m 時才認領該生活圈，否則視為無機能邊緣區
            if dist <= 1500.0 and nearest_road_node in road_node_to_fac_ids:
                fac_list_at_node = road_node_to_fac_ids[nearest_road_node]
                if fac_list_at_node:
                    cluster_id = post_partition.get(fac_list_at_node[0], -1)
                    
        geom_mean_post = (s1_p * s2_p * s3_p * s4_p) ** 0.25
        post_scores.append(geom_mean_post)
        assigned_clusters.append(cluster_id)
        
    df_res = pd.DataFrame({
        "Grid_ID": gdf_grids["Grid_ID"].values,
        "災前_防災韌性(幾何平均)": baseline_scores,
        "災後_防災韌性(幾何平均)": post_scores,
        "生活圈分群ID": assigned_clusters
    })
    df_res["最終韌性退化差值"] = df_res["災後_防災韌性(幾何平均)"] - df_res["災前_防災韌性(幾何平均)"]
    
    return df_res

# ==========================================
# 🏃‍♂️ 啟動單次空間模擬評估
# ==========================================
st.markdown("---")
st.subheader("🏁 第二步：啟動生活圈分群模擬與指標計算")

if st.button("🔥 執行單次空間失能評估"):
    with st.spinner(f"⏳ 正在進行拓樸分裂偵測與生活圈分群著色中..."):
        
        df_result = run_single_disaster_simulation(
            st.session_state["twd97_x"], 
            st.session_state["twd97_y"], 
            disaster_radius
        )
        
        st.success(f"🎉 模擬計算成功！成果已在下方輸出。")
        
        # ------------------------------------------
        # 📊 統計結果：以生活圈分群為單位
        # ------------------------------------------
        st.subheader("📊 各生活圈分群之災前與災後防災韌性統計總表")
        
        df_cluster_summary = df_result.groupby("生活圈分群ID").agg(
            包含網格總數=("Grid_ID", "count"),
            災前平均防災韌性=("災前_防災韌性(幾何平均)", "mean"),
            災後平均防災韌性=("災後_防災韌性(幾何平均)", "mean"),
            平均韌性退化差值=("最終韌性退化差值", "mean")
        ).reset_index()
        
        df_cluster_summary["生活圈分群ID"] = df_cluster_summary["生活圈分群ID"].apply(
            lambda x: "受災與無機能孤立區" if x == -1 else f"防衛生活圈 分群 {int(x)}"
        )
        
        st.dataframe(df_cluster_summary.style.format({
            "災前平均防災韌性": "{:.4f}",
            "災後平均防災韌性": "{:.4f}",
            "平均韌性退化差值": "{:.4f}"
        }), use_container_width=True)
        
        # ------------------------------------------
        # 🖼️ 繪製修正後的 Matplotlib 分群彩圖
        # ------------------------------------------
        st.subheader("🖼️ 全臺中市災後生活圈網絡分群彩圖")
        
        gdf_res_map = gdf_grids.merge(df_result, on="Grid_ID")
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        # 1. 先畫灰色底圖（無設施涵蓋的區塊）
        gdf_res_map.plot(ax=ax, color="#f5f5f5", edgecolor="none")
        
        # 2. 畫出受災與無機能孤立區（用淡紅色標註出來）
        gdf_isolated = gdf_res_map[gdf_res_map["生活圈分群ID"] == -1]
        if not gdf_isolated.empty:
            gdf_isolated.plot(ax=ax, color="#fcdad7", edgecolor="none")
            
        # 3. 畫出成功分裂出的多個彩色生活圈群集
        gdf_clustered = gdf_res_map[gdf_res_map["生活圈分群ID"] != -1]
        if not gdf_clustered.empty:
            gdf_clustered.plot(
                column="生活圈分群ID", 
                ax=ax, 
                categorical=True, 
                cmap="tab20", 
                edgecolor="none",
                legend=True,
                legend_kwds={'title': '防衛生活圈群集', 'loc': 'upper right', 'bbox_to_anchor': (1.25, 1)}
            )
            
        # 4. 畫出災害中心的紅色轟炸虛線圓圈
        disaster_circ = Point(st.session_state["twd97_x"], st.session_state["twd97_y"]).buffer(disaster_radius)
        gpd.GeoSeries([disaster_circ]).plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2, linestyle="--")
        
        # 標註災害中心點
        ax.scatter(st.session_state["twd97_x"], st.session_state["twd97_y"], color="red", marker="X", s=120, label="災害中心點")
        
        ax.set_title(f"臺中市災後防衛生活圈空間裂解分群圖\n(模擬半徑: {disaster_radius} 公尺)", fontsize=14, fontweight='bold')
        ax.set_xlabel("TWD97 X 座標 (m)")
        ax.set_ylabel("TWD97 Y 座標 (m)")
        ax.grid(True, linestyle=":", alpha=0.5)
        
        st.pyplot(fig)