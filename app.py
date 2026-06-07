import streamlit as st
import geopandas as gpd
import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import Point
# ... 匯入你原本的其他套件（如 scipy, community 等）

# 1. 設置網頁標題與介紹
st.title("臺中市都市防災空間網絡韌性評估系統")
st.markdown("可在左側面板或地圖上自訂災害中心與失能半徑，即時重算生活圈受災衝擊。")

# ==========================================
# 📥 資料載入快取 (避免每次點地圖都要重新下載 OSM 網格)
# ==========================================
@st.cache_data
def load_base_data():
    # 將原來的 data_folder 改為專案目錄下的相對路徑
    # 記得把村里人口、超商、收容所等檔案放到專案的 data/ 資料夾中
    data_dir = "data"
    
    # 下載 OSM 路網
    G_raw = ox.graph_from_place("Taichung, Taiwan", network_type="drive")
    G_proj = ox.project_graph(G_raw, to_crs="EPSG:3826")
    G_undirected = G_proj.to_undirected()
    
    # 讀取你的其他圖資... (醫院, 超商, 避難所)
    # 這裡放你原本 try-except 讀取各圖層的邏輯
    return G_undirected

G_undirected = load_base_data()

# ==========================================
# ⚙️ 側邊欄：讓使用者自訂條件 (滿足你的兩個條件)
# ==========================================
st.sidebar.header("災害情境設定")

# 條件二：自訂失能影響範圍 (幾公尺)
disaster_radius = st.sidebar.slider("請指定道路失能半徑 (公尺)", min_value=100, max_value=5000, value=3000, step=100)

# 條件一：自行輸入或點選失能點座標 (也可以整合 st_folium 讓滑鼠直接點)
st.sidebar.subheader("災害中心點座標 (TWD97)")
custom_x = st.sidebar.number_input("X 座標", value=217432)
custom_y = st.sidebar.number_input("Y 座標", value=2672145)

# ==========================================
# 🛠️ 核心演算法定義 (直接複製你寫好的功能)
# ==========================================
# 把你寫的：clean_and_project, map_facilities_to_roads, 
# run_custom_disaster_simulation 等函式原封不動貼在這裡。

# ==========================================
# 🏃‍♂️ 執行按鈕與結果呈現
# ==========================================
if st.sidebar.button("開始進行空間失能模擬"):
    with st.spinner("正在炸毀範圍內道路並重新計算生活圈... 請稍候..."):
        
        # 執行你寫好的運算
        sim_scores, sim_meta = run_custom_disaster_simulation(custom_x, custom_y, disaster_radius)
        
        # 這裡接你原本的結果彙整與計算抗災力差值 (Resilience Value) 的程式碼...
        
        # 呈現結果 (替代原本的 display)
        st.success("🎉 災害模擬完成！")
        
        st.subheader("📊 本次特定災害事件之空間數據紀錄")
        df_meta = pd.DataFrame([sim_meta])
        st.dataframe(df_meta) # 用 streamlit 的方式呈現表格
        
        st.subheader("🔥 空間網絡衝擊最嚴重的前 10 個網格")
        # 假設你的結果叫做 gdf_resilience_result
        st.dataframe(gdf_resilience_result[["Grid_ID", "災前_網格防災韌性綜合分數", "自訂災後分數", "最終韌性變化分數"]].sort_values(by="最終韌性變化分數").head(10))