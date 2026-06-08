import os
import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st

def calculate_perfect_scores(gdf_corridor_polygons, affected_cluster_id=None):
    """
    計算防災生活圈真實完美分數的函數 (Streamlit 優化版)
    """
    st.info("【計量地理統計】正在將各項防災指標分數無縫封裝至『真實全連通路網面』的屬性表...")

    # =====================================================================
    # 0. 前置作業：確保輸入的圖層有正確的欄位名稱與格式
    # =====================================================================
    # 複製一份避免更動到原始資料
    gdf_corridor = gdf_corridor_polygons.copy()
    
    # 【關鍵修復】檢查並確保 cluster_id 存在於欄位中（不讓它留在 index 或變成大寫）
    if "cluster_id" not in gdf_corridor.columns:
        if gdf_corridor.index.name == "cluster_id":
            gdf_corridor = gdf_corridor.reset_index()
        elif "CLUSTER_ID" in gdf_corridor.columns:
            gdf_corridor = gdf_corridor.rename(columns={"CLUSTER_ID": "cluster_id"})
        elif "Cluster_ID" in gdf_corridor.columns:
            gdf_corridor = gdf_corridor.rename(columns={"Cluster_ID": "cluster_id"})
        else:
            # 如果真的都沒有，強制建立一個，避免後面 groupby 噴 KeyError
            st.warning("⚠️ 找不到 cluster_id 欄位，將自動建立預設 ID。")
            gdf_corridor["cluster_id"] = gdf_corridor.index

    # =====================================================================
    # 1. 空間幾何交集：拿全連通面去切村里底圖 (vill.shp)，精算面積權重拆分人口
    # =====================================================================
    # 雲端佈署路徑防呆：先檢查相對路徑，若找不到再找絕對路徑
    vill_path = os.path.join("資料", "Vill_2.shp")
    if not os.path.exists(vill_path):
        # 備用路徑（配合妳同學原本的本機路徑或 Streamlit 根目錄）
        vill_path = "Vill_2.shp" 
        
    if not os.path.exists(vill_path):
        st.error(f"❌ 找不到村里底圖檔案：{vill_path}，請確認檔案已上傳至 GitHub 倉庫中。")
        return gdf_corridor

    gdf_v_pop = gpd.read_file(vill_path)
    
    # 確保兩邊座標系統一致為 EPSG:3826 (六度分帶)，才能精準計算面積 (平方公尺)
    if gdf_v_pop.crs != "EPSG:3826":
        gdf_v_pop = gdf_v_pop.to_crs("EPSG:3826")
    if gdf_corridor.crs != "EPSG:3826":
        gdf_corridor = gdf_corridor.to_crs("EPSG:3826")

    gdf_v_pop["村里總面積"] = gdf_v_pop.geometry.area

    st.write("   - [分母計算] 正在執行空間交集切割 (Overlay Intersection)...")
    
    # 進行空間交集
    intersections = gpd.overlay(gdf_corridor, gdf_v_pop, how="intersection")
    intersections["碎片交集面積"] = intersections.geometry.area

    # 核心公式：碎片人口 = 該里總人口(total) * (該生活圈佔該里的碎片面積 / 該里總面積)
    intersections["碎片分配人口"] = intersections["total"] * (
        intersections["碎片交集面積"] / intersections["村里總面積"]
    )

    # 加總得到每個生活圈的「真實總人口分母」
    df_cluster_pop_perfect = intersections.groupby("cluster_id")["碎片分配人口"].sum().reset_index()
    df_cluster_pop_perfect.columns = ["cluster_id", "生活圈真實總人口_分母"]

    # =====================================================================
    # 2. 分子計算：統計各生活圈內各類機能點的數量與避難收容所容量
    # =====================================================================
    # 註：這裡假設 road_node_to_facilities 與 road_node_to_cluster 存在於全域變數中
    # 如果妳這兩個變數是在外部定義的，Streamlit 會自動去抓取
    all_fac_rows = []
    
    # 防呆：確保外部字典存在
    try:
        for road_node, fac_list in road_node_to_facilities.items():
            c_id = road_node_to_cluster.get(road_node, -1)
            for fac in fac_list:
                all_fac_rows.append({
                    "cluster_id": c_id,
                    "type": fac["facility_type"],
                    "indoor_capacity": float(fac["data_row"].get("室內人數", 0) or 0)
                })
    except NameError:
        st.error("❌ 找不到 `road_node_to_facilities` 或 `road_node_to_cluster` 字典，請確保前半段程式碼已正確執行。")
        return gdf_corridor

    df_fac_all = pd.DataFrame(all_fac_rows)
    
    if df_fac_all.empty:
        st.error("❌ 統計出的機能點資料為空，請檢查設施資料讀取是否正常。")
        return gdf_corridor

    # 統計全台中各類資源的「全域總點數」，作為均權分母
    global_counts = df_fac_all.groupby("type").size().to_dict()

    # 統計各生活圈內部的各類資源數量
    df_cluster_counts = df_fac_all.groupby(["cluster_id", "type"]).size().unstack(fill_value=0).reset_index()

    # 獨立計算避難收容所的【室內人數總和分子】
    if "避難收容所" in df_fac_all["type"].values:
        df_indoor_sum = df_fac_all[df_fac_all["type"] == "避難收容所"].groupby("cluster_id")["indoor_capacity"].sum().reset_index()
    else:
        df_indoor_sum = pd.DataFrame(columns=["cluster_id", "indoor_capacity"])
        
    df_indoor_sum.columns = ["cluster_id", "生活圈總室內人數_分子"]

    # =====================================================================
    # 3. ✨核心公式計分✨：將所有因子分數塞進 Dataframe
    # =====================================================================
    df_scores_calc = (
        df_cluster_counts
        .merge(df_indoor_sum, on="cluster_id", how="left")
        .merge(df_cluster_pop_perfect, on="cluster_id", how="left")
        .fillna(0)
    )

    # 執行四大類別的「均權正規化分數 (1 / 全域總數)」
    for col in ["醫院", "五大超商", "量販店", "加油站"]:
        if col in df_scores_calc.columns and col in global_counts and global_counts[col] > 0:
            df_scores_calc[f"{col}_因子分數"] = df_scores_calc[col] / global_counts[col]
        else:
            df_scores_calc[f"{col}_因子分數"] = 0.0

    # 計算避難收容所供需比分數，並強制封頂在 1.5 倍，防止深山假性高分
    if "生活圈真實總人口_分母" in df_scores_calc.columns and "生活圈總室內人數_分子" in df_scores_calc.columns:
        shelter_ratio = np.where(
            df_scores_calc["生活圈真實總人口_分母"] > 0,
            df_scores_calc["生活圈總室內人數_分子"] / df_scores_calc["生活圈真實總人口_分母"],
            0.0
        )
    else:
        shelter_ratio = 0.0
        
    df_scores_calc["避難收容所_因子分數"] = np.clip(shelter_ratio, a_min=0.0, a_max=1.5)

    # =====================================================================
    # 4. 🛠️ 屬性封裝：將所有分數合流回我們的真實馬路連通面 (GeoDataFrame)
    # =====================================================================
    cols_to_drop = ["醫院_因子分數", "五大超商_因子分數", "量販店_因子分數", "加油站_因子分數", "避難收容所_因子分數", "生活圈防災機能總分數"]
    gdf_final_output = gdf_corridor.drop(columns=[c for c in cols_to_drop if c in gdf_corridor.columns])

    # 無縫 Merge 核心分數
    gdf_final_output = gdf_final_output.merge(
        df_scores_calc[[
            "cluster_id", 
            "醫院_因子分數", 
            "五大超商_因子分數", 
            "量販店_因子分數", 
            "加油站_因子分數", 
            "避難收容所_因子分數"
        ]], 
        on="cluster_id", 
        how="left"
    ).fillna(0)

    # 更名確保清楚
    if "生活圈跨區_Closeness" in gdf_final_output.columns:
        gdf_final_output = gdf_final_output.rename(columns={"生活圈跨區_Closeness": "跨區_Closeness_權重"})
    
    # 防呆：如果原本就沒有這個權重欄位，給予預設值 0
    if "跨區_Closeness_權重" not in gdf_final_output.columns:
        gdf_final_output["跨區_Closeness_權重"] = 0.0

    # =====================================================================
    # 【升級版修正】：特徵縮放 (Min-Max Normalization) + 幾何平均計分
    # =====================================================================
    def min_max_norm(series):
        if series.max() == series.min():
            return pd.Series(0.0, index=series.index)
        return (series - series.min()) / (series.max() - series.min())

    gdf_final_output["醫院_Norm"] = min_max_norm(gdf_final_output["醫院_因子分數"])
    gdf_final_output["五大超商_Norm"] = min_max_norm(gdf_final_output["五大超商_因子分數"])
    gdf_final_output["量販店_Norm"] = min_max_norm(gdf_final_output["量販店_因子分數"])
    gdf_final_output["加油站_Norm"] = min_max_norm(gdf_final_output["加油站_因子分數"])
    gdf_final_output["避難收容_Norm"] = min_max_norm(gdf_final_output["避難收容所_因子分數"])
    gdf_final_output["Closeness_Norm"] = min_max_norm(gdf_final_output["跨區_Closeness_權重"])

    # --- 方案 B：破解木桶效應的幾何平均法 (Geometric Mean) ---
    eps = 0.01
    gdf_final_output["生活圈防災機能總分數"] = (
        (gdf_final_output["醫院_Norm"] + eps) *
        (gdf_final_output["避難收容_Norm"] + eps) *
        ((gdf_final_output["五大超商_Norm"] + gdf_final_output["量販店_Norm"] + gdf_final_output["加油站_Norm"])/3 + eps) *
        (gdf_final_output["Closeness_Norm"] + eps)
    ) ** (1/4) * 100 

    # =====================================================================
    # 5. 💾 存檔釋放：導出成 Shapefile 網格面圖層
    # =====================================================================
    gdf_shp_save = gdf_final_output.rename(columns={
        "跨區_Closeness_權重": "Cc_weight",
        "醫院_因子分數": "F_hospital",
        "五大超商_因子分數": "F_store",
        "量販店_因子分數": "F_market",
        "加油站_因子分數": "F_gas",
        "避難收容所_因子分數": "F_shelter",
        "生活圈防災機能總分數": "TotalScore"
    })

    # Streamlit 雲端儲存路徑防呆
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        gdf_shp_save.to_file(os.path.join(output_folder, "臺中市防災生活圈_真實路網量化評分網格.shp"), encoding="utf-8")
        st.success("🎉 【成功】地理網格圖層已成功導出存檔為 'output/臺中市防災生活圈_真實路網量化評分網格.shp'！")
    except Exception as e:
        st.warning(f"⚠️ Shapefile 儲存失敗 (可能是雲端權限問題)，但不影響畫面顯示。錯誤訊息: {e}")

    # 回傳計算好的完整 GeoDataFrame，供 Streamlit 後續畫地圖使用
    return gdf_final_output