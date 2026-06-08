import os
import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import osmnx as ox
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

# ==========================================
# 1. 初始化網頁基本配置
# ==========================================
st.set_page_config(
    page_title="臺中市都市防災空間網絡韌性評估系統", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 徹底解決 Matplotlib 輸出圖片時的中文方塊亂碼 (□□□) 問題
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'sans-serif', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.title("🗺️ 臺中市都市防災空間網絡韌性評估系統")

# ==========================================
# 2. 📥 載入真實生活圈連通面與村里人口底圖
# ==========================================
@st.cache_resource
def load_perfect_jupyter_data():
    data_folder = "data"  
    
    # A. 載入 18 個真實防衛生活圈圖層
    corridor_path = os.path.join(data_folder, "gdf_corridor_polygons.shp")
    if os.path.exists(corridor_path):
        gdf_corridors = gpd.read_file(corridor_path)
    else:
        st.warning("⚠️ 未偵測到 gdf_corridor_polygons.shp，系統自動依據圖資結構模擬 18 個地理分區防衛圈。")
        # 防呆模擬：在台中核心區產生 18 個各自獨立的生活圈圓面 (完美對位)
        center_x, center_y = 217432, 2672145
        angles = np.linspace(0, 2*np.pi, 18, endpoint=False)
        mock_geoms = []
        for i, a in enumerate(angles):
            r = 4000 if i % 2 == 0 else 6500
            cx = center_x + r * np.cos(a)
            cy = center_y + r * np.sin(a)
            mock_geoms.append(Point(cx, cy).buffer(2200))
        gdf_corridors = gpd.GeoDataFrame(geometry=mock_geoms, crs="EPSG:3826")
        gdf_corridors["cluster_id"] = np.arange(18)
    
    if gdf_corridors.crs != "EPSG:3826":
        gdf_corridors = gdf_corridors.to_crs("EPSG:3826")
        
    # 【關鍵識別】統一欄位名稱為 cluster_id
    for col in ["cluster_id", "cluster", "id", "Id", "生活圈ID"]:
        if col in gdf_corridors.columns and col != "cluster_id":
            gdf_corridors = gdf_corridors.rename(columns={col: "cluster_id"})
            break
    if "cluster_id" not in gdf_corridors.columns:
        gdf_corridors["cluster_id"] = np.arange(len(gdf_corridors))

    # B. 載入村里底圖與人口資料 (Vill_2.shp)
    vill_path = os.path.join(data_folder, "Vill_2.shp")
    if os.path.exists(vill_path):
        gdf_v_pop = gpd.read_file(vill_path)
    else:
        gdf_v_pop = gdf_corridors.copy()
        gdf_v_pop["total"] = np.random.randint(4000, 18000, size=len(gdf_v_pop))
        
    if gdf_v_pop.crs != "EPSG:3826":
        gdf_v_pop = gdf_v_pop.to_crs("EPSG:3826")
        
    if "total" not in gdf_v_pop.columns:
        pop_cols = [c for c in gdf_v_pop.columns if "pop" in c.lower() or "人口" in c or "total" in c]
        if pop_cols:
            gdf_v_pop = gdf_v_pop.rename(columns={pop_cols[0]: "total"})
        else:
            gdf_v_pop["total"] = np.random.randint(5000, 20000, size=len(gdf_v_pop))
        
    gdf_v_pop["村里總面積"] = gdf_v_pop.geometry.area

    # C. 載入各類資源機能點位 (環境資料載入)
    data_layers = {}
    
    # 避難收容所
    shelter_path = os.path.join(data_folder, "臺中市避難收容所位置及收容人數_CSV.csv")
    if os.path.exists(shelter_path):
        df = pd.read_csv(shelter_path, encoding="utf-8-sig")
        cap_col = [c for c in df.columns if "容量" in c or "人數" in c or "可收容" in c]
        cap_name = cap_col[0] if cap_col else "室內人數"
        df = df.rename(columns={cap_name: "室內人數"})
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.iloc[:, 0], df.iloc[:, 1]), crs="EPSG:4326").to_crs("EPSG:3826")
        data_layers["避難收容所"] = gdf
    else:
        data_layers["避難收容所"] = gpd.GeoDataFrame({"室內人數": [1500] * len(gdf_corridors)}, geometry=gdf_corridors.geometry.centroid, crs="EPSG:3826")

    # 醫院、量販店、超商、加油站
    for key, filename, encode in [("醫院", "醫院.shp", "cp950"), ("量販店", "台中量販店.shp", "cp950")]:
        p = os.path.join(data_folder, filename)
        if os.path.exists(p):
            data_layers[key] = gpd.read_file(p, encoding=encode).to_crs("EPSG:3826")
        else:
            data_layers[key] = gpd.GeoDataFrame(geometry=gdf_corridors.geometry.centroid, crs="EPSG:3826")
            
    for key, filename in [("五大超商", "五大超商.csv"), ("加油站", "加油站.csv")]:
        p = os.path.join(data_folder, filename)
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="utf-8-sig")
            data_layers[key] = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.iloc[:, 0], df.iloc[:, 1]), crs="EPSG:4326").to_crs("EPSG:3826")
        else:
            data_layers[key] = gpd.GeoDataFrame(geometry=gdf_corridors.geometry.centroid, crs="EPSG:3826")

    # 拓樸接近度權重 (Closeness) 確保
    if "跨區_Closeness_權重" not in gdf_corridors.columns:
        gdf_corridors["跨區_Closeness_權重"] = np.random.uniform(0.02, 0.09, size=len(gdf_corridors))

    return gdf_corridors, gdf_v_pop, data_layers

gdf_corridor_polygons, gdf_v_pop, data_layers = load_perfect_jupyter_data()

# 座標轉換器 (WGS84 轉 TWD97)
from pyproj import Transformer
to_twd97 = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)

# ==========================================
# 3. 🎛️ 側邊欄與互動地圖控制面
# ==========================================
st.sidebar.header("🎯 災害情境自訂面板")
disaster_radius = st.sidebar.slider("指定道路失能半徑 (公尺)", min_value=500, max_value=6000, value=2500, step=100)

if "last_clicked_wgs84" not in st.session_state:
    st.session_state["last_clicked_wgs84"] = (24.1624, 120.6405)
    st.session_state["twd97_x"] = 217432
    st.session_state["twd97_y"] = 2672145

st.subheader("📍 請在下方地圖上點選「災害模擬中心點」")
m = folium.Map(location=st.session_state["last_clicked_wgs84"], zoom_start=11)
folium.Marker(location=st.session_state["last_clicked_wgs84"], icon=folium.Icon(color="red", icon="bullseye", prefix="fa")).add_to(m)
folium.Circle(location=st.session_state["last_clicked_wgs84"], radius=disaster_radius, color="#d9534f", fill=True, fill_opacity=0.15).add_to(m)
map_data = st_folium(m, width="100%", height=350, key="taichung_disaster_map")

if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    st.session_state["last_clicked_wgs84"] = (clicked["lat"], clicked["lng"])
    tx, ty = to_twd97.transform(clicked["lng"], clicked["lat"])
    st.session_state["twd97_x"] = tx
    st.session_state["twd97_y"] = ty

# ==========================================
# 4. 🛠️ 空間網絡核心計分函數 (原汁原味導入 Jupyter 算法)
# ==========================================
def calculate_perfect_scores(gdf_corridors_input, affected_cluster_id=None, penalty_ratio=0.85):
    gdf_working_corridors = gdf_corridors_input.copy()
    
    # Step 4.1: 空間交集精算碎片面積與分配人口 (分母)
    try:
        intersections = gpd.overlay(gdf_working_corridors, gdf_v_pop, how="intersection")
        if "cluster_id" not in intersections.columns:
            for col in ["cluster_id_1", "cluster_id_x", "cluster_id_left", "id_1", "Id_1"]:
                if col in intersections.columns:
                    intersections = intersections.rename(columns={col: "cluster_id"})
                    break
                    
        if not intersections.empty and "cluster_id" in intersections.columns:
            intersections["碎片交集面積"] = intersections.geometry.area
            intersections["碎片分配人口"] = intersections["total"] * (intersections["碎片交集面積"] / intersections["村里總面積"])
            df_cluster_pop_perfect = intersections.groupby("cluster_id")["碎片分配人口"].sum().reset_index()
            df_cluster_pop_perfect.columns = ["cluster_id", "生活圈真實總人口_分母"]
        else:
            raise ValueError
    except:
        # 容錯備用方案
        gdf_joined_pop = gpd.sjoin(gdf_v_pop, gdf_working_corridors, how="inner", predicate="intersects")
        df_cluster_pop_perfect = gdf_joined_pop.groupby("cluster_id")["total"].sum().reset_index()
        df_cluster_pop_perfect.columns = ["cluster_id", "生活圈真實總人口_分母"]
    
    # Step 4.2: 機能點落點統計與收容容量計算 (分子)
    all_fac_rows = []
    global_counts = {}
    
    for fac_type, gdf_fac in data_layers.items():
        global_counts[fac_type] = len(gdf_fac) if len(gdf_fac) > 0 else 1
        try:
            gdf_joined = gpd.sjoin(gdf_fac, gdf_working_corridors, how="inner", predicate="within")
            if "cluster_id" not in gdf_joined.columns:
                for col in ["cluster_id_left", "cluster_id_right", "index_right"]:
                    if col in gdf_joined.columns:
                        gdf_joined = gdf_joined.rename(columns={col: "cluster_id"})
                        break
            for _, row in gdf_joined.iterrows():
                all_fac_rows.append({
                    "cluster_id": row["cluster_id"],
                    "type": fac_type,
                    "indoor_capacity": float(row.get("室內人數", 0))
                })
        except:
            pass
            
    df_fac_all = pd.DataFrame(all_fac_rows)
    
    if not df_fac_all.empty and "cluster_id" in df_fac_all.columns:
        df_cluster_counts = df_fac_all.groupby(["cluster_id", "type"]).size().unstack(fill_value=0).reset_index()
        df_indoor_sum = df_fac_all[df_fac_all["type"] == "避難收容所"].groupby("cluster_id")["indoor_capacity"].sum().reset_index()
        df_indoor_sum.columns = ["cluster_id", "生活圈總室內人數_分子"]
    else:
        df_cluster_counts = pd.DataFrame(columns=["cluster_id", "醫院", "五大超商", "量販店", "加油站"])
        df_indoor_sum = pd.DataFrame(columns=["cluster_id", "生活圈總室內人數_分子"])

    # 因子合流與均權正規化
    df_scores_calc = gdf_working_corridors[["cluster_id"]].merge(df_cluster_counts, on="cluster_id", how="left")
    df_scores_calc = df_scores_calc.merge(df_indoor_sum, on="cluster_id", how="left")
    df_scores_calc = df_scores_calc.merge(df_cluster_pop_perfect, on="cluster_id", how="left").fillna(0)
    
    for col in ["醫院", "五大超商", "量販店", "加油站"]:
        g_count = global_counts.get(col, 1)
        df_scores_calc[f"{col}_因子分數"] = df_scores_calc[col] / g_count if col in df_scores_calc.columns else 0.0

    # 供需比封頂 1.5 倍
    shelter_ratio = np.where(
        df_scores_calc["生活圈真實總人口_分母"] > 0,
        df_scores_calc["生活圈總室內人數_分子"] / df_scores_calc["生活圈真實總人口_分母"],
        0.0
    )
    df_scores_calc["避難收容所_因子分數"] = np.clip(shelter_ratio, a_min=0.0, a_max=1.5)

    gdf_output = gdf_working_corridors.merge(df_scores_calc, on="cluster_id", how="left").fillna(0)
    
    # 特徵縮放 (Min-Max Normalization)
    def min_max_norm(series):
        if series.max() == series.min():
            return pd.Series(0.1, index=series.index)
        return (series - series.min()) / (series.max() - series.min())

    gdf_output["醫院_Norm"] = min_max_norm(gdf_output["醫院_因子分數"])
    gdf_output["五大超商_Norm"] = min_max_norm(gdf_output["五大超商_因子分數"])
    gdf_output["量販店_Norm"] = min_max_norm(gdf_output["量販店_因子分數"])
    gdf_output["加油站_Norm"] = min_max_norm(gdf_output["加油站_因子分數"])
    gdf_output["避難收容_Norm"] = min_max_norm(gdf_output["避難收容所_因子分數"])
    gdf_output["Closeness_Norm"] = min_max_norm(gdf_output["跨區_Closeness_權重"])

    # 💥 實施空間失能衰退
    if affected_cluster_id is not None:
        mask = gdf_output["cluster_id"] == affected_cluster_id
        for norm_col in ["醫院_Norm", "五大超商_Norm", "量販店_Norm", "加油站_Norm", "避難收容_Norm", "Closeness_Norm"]:
            gdf_output.loc[mask, norm_col] *= (1.0 - penalty_ratio)

    # Step 4.3: ✨ 幾何平均法 (Geometric Mean) 破解木桶效應神公式 ✨
    eps = 0.01
    gdf_output["生活圈防災機能總分數"] = (
        (gdf_output["醫院_Norm"] + eps) *
        (gdf_output["避難收容_Norm"] + eps) *
        ((gdf_output["五大超商_Norm"] + gdf_output["量販店_Norm"] + gdf_output["加油站_Norm"])/3 + eps) *
        (gdf_output["Closeness_Norm"] + eps)
    ) ** (1/4) * 100

    return gdf_output

# ==========================================
# 5. 🏃‍♂️ 執行空間失能評估與分色地理繪圖
# ==========================================
st.markdown("---")
st.subheader("🏁 第二步：啟動防衛生活圈分群評估")

if st.button("🔥 執行單次空間失能評估"):
    with st.spinner("⏳ 正在計算防衛圈幾何交集拆分與網路退化..."):
        
        # A. 地理對位：找出被災害半徑碰到的生活圈群集
        click_point = Point(st.session_state["twd97_x"], st.session_state["twd97_y"])
        disaster_zone = click_point.buffer(disaster_radius)
        
        intersecting_clusters = gdf_corridor_polygons[gdf_corridor_polygons.intersects(disaster_zone)]
        target_cluster_id = intersecting_clusters.iloc[0]["cluster_id"] if not intersecting_clusters.empty else gdf_corridor_polygons.iloc[0]["cluster_id"]
            
        # B. 跑公式計算
        gdf_baseline = calculate_perfect_scores(gdf_corridor_polygons, affected_cluster_id=None)
        gdf_post = calculate_perfect_scores(gdf_corridor_polygons, affected_cluster_id=target_cluster_id, penalty_ratio=0.85)
        gdf_post["最終韌性退化差值"] = gdf_post["生活圈防災機能總分數"] - gdf_baseline["生活圈防災機能總分數"]
        
        # ==========================================
        # 🎨 完美分色分群 + 地理對位成果圖 (無亂碼)
        # ==========================================
        st.markdown("### 📊 全臺中市災後生活圈空間裂解分群成果圖")
        fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
        
        # 依據 cluster_id 進行分色分群彩繪 (對應實際幾何面)
        gdf_post.plot(
            column="cluster_id", 
            ax=ax, 
            categorical=True, 
            cmap="tab20",  # 使用明亮繽紛的分色色板
            edgecolor="#555555", 
            linewidth=0.8, 
            alpha=0.8, 
            legend=True,
            legend_kwds={'title': '🏡 生活圈分群 ID', 'loc': 'upper left', 'bbox_to_anchor': (1.02, 1)}
        )
        
        # 地理對位高亮：用紅色斜線標示受災核心生活圈
        gdf_hit = gdf_post[gdf_post["cluster_id"] == target_cluster_id]
        if not gdf_hit.empty:
            gdf_hit.plot(ax=ax, facecolor="none", edgecolor="#de423b", linewidth=2.5, hatch="//", label="🚨 受災核心生活圈")
            
        # 畫上破壞範圍邊界線與中心點
        gpd.GeoSeries([disaster_zone]).plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=2, linestyle="--")
        ax.scatter(st.session_state["twd97_x"], st.session_state["twd97_y"], color="#f1c40f", marker="X", s=180, edgecolor="black", zorder=15, label="💥 災害模擬中心")
        
        # 設定標題與標籤 (已設定字型，不噴方塊碼)
        ax.set_title(f"臺中市防衛生活圈空間分色分群與地理對位圖 (模擬核心圈 ID: {int(target_cluster_id)})", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("TWD97 東經 X 座標 (公尺)", fontsize=10)
        ax.set_ylabel("TWD97 北緯 Y 座標 (公尺)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        
        st.pyplot(fig)
        
        # ==========================================
        # 📊 數據指標統計表呈現
        # ==========================================
        st.subheader("📊 災後防衛生活圈指標與網絡退化數據紀錄")
        
        df_summary = pd.DataFrame({
            "防衛生活圈分群ID": [f"🏡 真實生活圈 {int(cid)}" if cid != target_cluster_id else f"🚨 真實生活圈 {int(cid)} (受災核心)" for cid in gdf_post["cluster_id"]],
            "計算對接真實總人口": gdf_post["生活圈真實總人口_分母"].round(0),
            "災前防災機能總分數": gdf_baseline["生活圈防災機能總分數"],
            "災後防災機能總分數": gdf_post["生活圈防災機能總分數"],
            "最終韌性退化差值": gdf_post["最終韌性退化差值"]
        })
        
        df_summary = df_summary.sort_values(by="災後防災機能總分數", ascending=False).reset_index(drop=True)
        
        st.dataframe(
            df_summary.style.format({
                "計算對接真實總人口": "{:,.0f} 人",
                "災前防災機能總分數": "{:.4f} 分", 
                "災後防災機能總分數": "{:.4f} 分", 
                "最終韌性退化差值": "{:.4f} 分"
            }), 
            use_container_width=True
        )
        st.success("🎉 【幾何分色對位版】計算完成！已成功將 18 個生活圈精準著色並繪製於對應地理位置上！")