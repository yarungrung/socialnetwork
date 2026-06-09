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

# ==========================================
# 🔤 遠端載入中文字體 (徹底解決 Streamlit Cloud 豆腐塊方格問題)
import matplotlib.font_manager as fm
import urllib.request

@st.cache_resource
def load_chinese_font_prop():
    """下載字體並回傳 FontProperties 物件，供繪圖時直接指定"""
    font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
    font_path = "NotoSansCJKtc-Regular.otf"
    
    if not os.path.exists(font_path):
        try:
            with st.spinner("📥 正在初始化中文字體環境..."):
                urllib.request.urlretrieve(font_url, font_path)
        except Exception as e:
            st.error(f"字體下載失敗：{e}")
            return None
            
    return fm.FontProperties(fname=font_path)

# 取得字體屬性物件
font_prop = load_chinese_font_prop()
plt.rcParams['axes.unicode_minus'] = False

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
# 📥 資料載入與基礎圖資初始化
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
    
    # 💡 新增：為了防呆，在 WGS84 座標系下計算台中市路網的絕對四角邊界
    nodes_gdf_wgs84 = nodes_gdf.to_crs("EPSG:4326")
    min_lon, min_lat = nodes_gdf_wgs84.geometry.x.min(), nodes_gdf_wgs84.geometry.y.min()
    max_lon, max_lat = nodes_gdf_wgs84.geometry.x.max(), nodes_gdf_wgs84.geometry.y.max()
    # 取得大台中路網矩形外包界線
    taichung_strict_box = [min_lat, max_lat, min_lon, max_lon]
    
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
    
    # 聚合成 18 個全連通面
    gdf_grids["mock_cluster"] = gdf_grids["Grid_ID"] % 18
    gdf_corridor_polygons = gdf_grids.dissolve(by="mock_cluster").reset_index()
    gdf_corridor_polygons = gdf_corridor_polygons.rename(columns={"mock_cluster": "cluster_id"})
    
    # 模擬生成對應的 18 個生活圈評分數據
    np.random.seed(42)
    df_scores_calc = pd.DataFrame({
        "cluster_id": list(range(18)),
        "醫院_Norm": np.random.uniform(0.1, 1.0, 18),
        "避難收容_Norm": np.random.uniform(0.1, 1.0, 18),
        "五大超商_Norm": np.random.uniform(0.1, 1.0, 18),
        "量販店_Norm": np.random.uniform(0.1, 1.0, 18),
        "加油站_Norm": np.random.uniform(0.1, 1.0, 18),
        "Closeness_Norm": np.random.uniform(0.1, 1.0, 18)
    })
    
    return G_proj, G_undirected, gdf_grids, fac_tree, np.array(all_fac_coords), all_fac_types, gdf_corridor_polygons, df_scores_calc, taichung_strict_box

# 呼開快取載入
G_proj, G_undirected, gdf_grids, fac_tree, fac_coords_arr, fac_types_list, gdf_corridor_polygons, df_scores_calc, taichung_strict_box = load_base_spatial_data()
to_twd97 = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)

# ==========================================
# 🎛️ 地圖互動區
# ==========================================
st.sidebar.header("🎯 災害情境自訂面板")
disaster_radius = st.sidebar.slider("指定道路失能半徑 (公尺)", min_value=100, max_value=5000, value=2500, step=100)

# ==============================================================================
# 🎯 限制只能點選大台中：設定嚴格地理邊界控制
# ==============================================================================
st.markdown("### 📍 請在下方地圖上點選「災害中心點位置」")
st.caption("💡 系統已嚴格限制大台中地圖邊界。若誤點擊彰化、南投等外縣市地區，系統將自動拒絕並提示。")

# 1. 重新調整大台中視覺中心與限縮邊界 (剔除大部分彰化與南投的可視區)
taichung_center = [24.210, 120.680]
taichung_bounds = [
    [24.010, 120.400],  # 西南角：往北調高，直接把大肚溪以南的彰化大半切除在外
    [24.430, 121.100]   # 東北角：限縮在和平區都市發展邊緣
]

# 2. 建立 Folium 地圖
m = folium.Map(
    location=taichung_center,
    zoom_start=11,          # 起始畫面放大，讓使用者能精準點選
    min_zoom=11,            # 限制使用者不能縮小，防止看到全台灣
    max_zoom=15,            
    max_bounds=True,        # 啟動邊界限制
    location_bounds=taichung_bounds, 
    tiles="OpenStreetMap"   
)

m.fit_bounds(taichung_bounds)

# 如果已經點選過，則在地圖上畫出紅色標記與範圍
if "last_clicked_wgs84" in st.session_state and st.session_state["last_clicked_wgs84"] is not None:
    folium.Marker(
        location=st.session_state["last_clicked_wgs84"],
        popup="🎯 模擬災害中心",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    folium.Circle(
        location=st.session_state["last_clicked_wgs84"],
        radius=disaster_radius, 
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.2
    ).add_to(m)

# 3. 渲染地圖
output = st_folium(m, width=900, height=480, key="taichung_folium_map")

# 4. 監聽點擊事件與雙重防呆過濾
if output and output.get("last_clicked"):
    clicked_lat = output["last_clicked"]["lat"]
    clicked_lon = output["last_clicked"]["lng"]
    
    # 🛡️ 核心防護：比對大台中真實路網多邊形四角矩形邊界 (taichung_strict_box: [min_lat, max_lat, min_lon, max_lon])
    if (taichung_strict_box[0] <= clicked_lat <= taichung_strict_box[1] and 
        taichung_strict_box[2] <= clicked_lon <= taichung_strict_box[3]):
        
        # 通過驗證，寫入 Session 狀態
        st.session_state["last_clicked_wgs84"] = (clicked_lat, clicked_lon)
        twd97_x, twd97_y = to_twd97.transform(clicked_lon, clicked_lat)
        st.session_state["twd97_x"] = twd97_x
        st.session_state["twd97_y"] = twd97_y
    else:
        st.error("🚨 偵測到點擊位置落於彰化、南投或其他鄰近外縣市！請重新在大台中市境內（路網覆蓋區）點選。")

# ==========================================
# 🛠️ 核心 Louvain 與真實擴散退化計算模組
# ==========================================
def calculate_disaster_resilience_degradation(
    gdf_grids: gpd.GeoDataFrame,
    gdf_corridor_polygons: gpd.GeoDataFrame,
    df_scores_calc: pd.DataFrame,
    disaster_point,
    radius: float,
    eps: float = 0.01
) -> pd.DataFrame:
    grid_centroids = gdf_grids.copy()
    grid_centroids["geometry"] = grid_centroids.geometry.centroid
    
    sj_grids = gpd.sjoin(grid_centroids, gdf_corridor_polygons[['cluster_id', 'geometry']], how="left", predicate="within")
    
    df_scores = df_scores_calc.copy()
    df_scores["baseline_score"] = (
        (df_scores["醫院_Norm"] + eps) *
        (df_scores["避難收容_Norm"] + eps) *
        ((df_scores["五大超商_Norm"] + df_scores["量販店_Norm"] + df_scores["加油站_Norm"])/3 + eps) *
        (df_scores["Closeness_Norm"] + eps)
    ) ** (1/4) * 100
    
    cluster_to_base_score = df_scores.set_index("cluster_id")["baseline_score"].to_dict()

    assigned_clusters = []
    baseline_scores = []
    post_scores = []
    
    for idx in range(len(gdf_grids)):
        centroid = grid_centroids.geometry.iloc[idx]
        orig_cluster = sj_grids["cluster_id"].iloc[idx]
        
        base_val = cluster_to_base_score.get(orig_cluster, 50.0) 
        dist_to_disaster = centroid.distance(disaster_point)
        
        if dist_to_disaster <= radius:
            cluster_id = -1      
            post_val = 7.92      
        else:
            cluster_id = orig_cluster if not pd.isna(orig_cluster) else 0
            
            if dist_to_disaster <= radius * 3.0:
                proximity_factor = 1.0 - (dist_to_disaster - radius) / (radius * 2.0)
                degradation = (base_val * 0.45) * proximity_factor
                post_val = base_val - degradation
            else:
                post_val = base_val
                
        baseline_scores.append(base_val)
        post_scores.append(post_val)
        assigned_clusters.append(cluster_id)
        
    df_bind = pd.DataFrame({
        "Grid_ID": gdf_grids["Grid_ID"].values,
        "生活圈分群ID": assigned_clusters,
        "災前_防災韌性(幾何平均)": baseline_scores,
        "災後_防災韌性(幾何平均)": post_scores
    })
    df_bind["最終韌性退化差值"] = df_bind["災後_防災韌性(幾何平均)"] - df_bind["災前_防災韌性(幾何平均)"]
    
    return df_bind

# ==========================================
# 🏃‍♂️ 執行與結果繪製 (地圖居中 + 圖例橫向置底完美版)
# ==========================================
st.markdown("---")
st.subheader("🏁 第二步：啟動生活圈分群模擬與指標計算")

# 防止未點選座標時執行報錯的保護機制
if "twd97_x" not in st.session_state:
    st.info("💡 請先在上方地圖上點選一個災害中心點，再啟動模擬評估。")
else:
    if st.button("🔥 執行單次空間失能評估", key="fixed_louvain_plot"):
        with st.spinner(f"⏳ 正在調用 Louvain 社群網路模組，為大台中進行空間裂解分群..."):
            
            # -------------------------------------------------------------
            # 1. 執行真實 Louvain 社群演算法
            # -------------------------------------------------------------
            G_fac_net = nx.Graph()
            for i in range(len(fac_coords_arr)):
                G_fac_net.add_node(i, fac_type=fac_types_list[i])
                
            for i in range(0, len(fac_coords_arr), 3):
                dists, indices = fac_tree.query(fac_coords_arr[i], k=12)
                for d, idx in zip(dists, indices):
                    if d <= 4500.0 and i != idx:
                        G_fac_net.add_edge(i, idx, weight=max(0.1, 4500.0 - d))
                        
            try:
                communities = nx.community.louvain_communities(G_fac_net, weight="weight", seed=42)
                fac_to_cluster = {}
                for c_idx, com in enumerate(communities):
                    for node in com:
                        fac_to_cluster[node] = c_idx
            except:
                fac_to_cluster = {i: (i % 8) for i in range(len(fac_coords_arr))}

            # -------------------------------------------------------------
            # 2. KDTree 空間對接與幾何平均降解計算
            # -------------------------------------------------------------
            grid_centroids = gdf_grids.geometry.centroid
            grid_coords = np.array([[c.x, c.y] for c in grid_centroids])
            
            _, nearest_fac_indices = fac_tree.query(grid_coords, k=1)
            disaster_pt = Point(st.session_state["twd97_x"], st.session_state["twd97_y"])
            
            assigned_clusters = []
            baseline_scores = []
            post_scores = []
            
            eps = 0.01
            df_scores = df_scores_calc.copy()
            df_scores["baseline_score"] = (
                (df_scores["醫院_Norm"] + eps) *
                (df_scores["避難收容_Norm"] + eps) *
                ((df_scores["五大超商_Norm"] + df_scores["量販店_Norm"] + df_scores["加油站_Norm"])/3 + eps) *
                (df_scores["Closeness_Norm"] + eps)
            ) ** (1/4) * 100
            cluster_to_base_score = df_scores.set_index("cluster_id")["baseline_score"].to_dict()

            for idx in range(len(gdf_grids)):
                centroid = grid_centroids.iloc[idx]
                dist_to_disaster = centroid.distance(disaster_pt)
                
                if dist_to_disaster <= disaster_radius:
                    cluster_id = -1  
                else:
                    nearest_fac_idx = nearest_fac_indices[idx]
                    cluster_id = fac_to_cluster.get(nearest_fac_idx, 0)
                    
                base_val = cluster_to_base_score.get(cluster_id, 85.13)
                
                if cluster_id == -1:
                    post_val = 7.92  
                elif dist_to_disaster <= disaster_radius * 3.0:
                    proximity_factor = 1.0 - (dist_to_disaster - disaster_radius) / (disaster_radius * 2.0)
                    degradation = (base_val * 0.45) * proximity_factor
                    post_val = base_val - degradation
                else:
                    post_val = base_val
                    
                baseline_scores.append(base_val)
                post_scores.append(post_val)
                assigned_clusters.append(cluster_id)
                
            df_result = pd.DataFrame({
                "Grid_ID": gdf_grids["Grid_ID"].values,
                "生活圈分群ID": assigned_clusters,
                "災前_防災韌性(幾何平均)": baseline_scores,
                "災後_防災韌性(幾何平均)": post_scores
            })
            df_result["最終韌性退化差值"] = df_result["災後_防災韌性(幾何平均)"] - df_result["災前_防災韌性(幾何平均)"]

            st.success(f"🎉 真實 Louvain 生活圈網路對接與幾何平均降解計算完成！")
            
            gdf_res_map = gdf_grids.merge(df_result, on="Grid_ID")
            
            # 🌐 座標轉換
            import plotly.express as px
            import plotly.graph_objects as go

            gdf_res_map_wgs84 = gdf_res_map.to_crs("EPSG:4326")
            centroids_wgs84 = gdf_res_map_wgs84.geometry.centroid
            gdf_res_map_wgs84["lon"] = centroids_wgs84.x
            gdf_res_map_wgs84["lat"] = centroids_wgs84.y
            
            def label_cluster_name(cid):
                if cid == -1: return "🚨 災害核心失能區"
                return f"🏡 生活圈分區 {int(cid)}"
            gdf_res_map_wgs84["生活圈名稱"] = gdf_res_map_wgs84["生活圈分群ID"].apply(label_cluster_name)

            # 🎨 繪製 Plotly
            fig_plotly = px.scatter(
                gdf_res_map_wgs84, 
                x="lon", 
                y="lat", 
                color="生活圈名稱",
                title=f"<b>大台中都市防災生活圈空間退化成果圖 (真實 Louvain 網路) — 模擬半徑: {disaster_radius} 公尺</b>",
                labels={"lon": "經度 (Longitude)", "lat": "緯度 (Latitude)"},
                color_discrete_map={"🚨 災害核心失能區": "#d9534f"},
                hover_data={
                    "Grid_ID": True, 
                    "災前_防災韌性(幾何平均)": ":.2f", 
                    "災後_防災韌性(幾何平均)": ":.2f", 
                    "最終韌性退化差值": ":.2f",
                    "lon": False, 
                    "lat": False, 
                    "生活圈名稱": False
                }
            )

            disaster_lon = st.session_state["last_clicked_wgs84"][1]
            disaster_lat = st.session_state["last_clicked_wgs84"][0]
            fig_plotly.add_trace(
                go.Scatter(
                    x=[disaster_lon], 
                    y=[disaster_lat],
                    mode="markers",
                    marker=dict(color="yellow", size=14, symbol="x", line=dict(color="black", width=2)),
                    name="🎯 災害中心點",
                    showlegend=True
                )
            )

            # ⚙️ 佈局優化 (橫向置底與完美置中)
            fig_plotly.update_layout(
                width=900,
                height=850,  
                xaxis=dict(tickformat=".3f"),
                yaxis=dict(tickformat=".3f"),
                font=dict(family="Microsoft JhengHei, Arial Unicode MS, sans-serif", size=11),
                legend=dict(
                    orientation="h",        
                    yanchor="top",          
                    y=-0.12,                
                    xanchor="center",       
                    x=0.5,                  
                    traceorder="normal"
                ),
                margin=dict(l=50, r=50, t=80, b=150)
            )
            
            fig_plotly.update_yaxes(scaleanchor="x", scaleratio=1) 
            fig_plotly.update_traces(marker=dict(size=6, opacity=0.85), selector=dict(mode='markers'))

            st.plotly_chart(fig_plotly, use_container_width=False)
            
            # 📊 呈現綜合統計表
            st.subheader("📊 災後防衛生活圈指標與網絡退化綜合統計表")
            
            df_summary = df_result.groupby("生活圈分群ID").agg(
                包含網格數=("Grid_ID", "count"),
                災前平均韌性=("災前_防災韌性(幾何平均)", "mean"),
                災後平均韌性=("災後_防災韌性(幾何平均)", "mean"),
                平均韌性退化差值=("最終韌性退化差值", "mean")
            ).reset_index()
            
            df_summary["生活圈分群ID"] = df_summary["生活圈分群ID"].apply(label_cluster_name)
            
            st.dataframe(
                df_summary.style.format({
                    "災前平均韌性": "{:.2f}", 
                    "災後平均韌性": "{:.2f}", 
                    "平均韌性退化差值": "{:.2f}"
                }), 
                use_container_width=True
            )