import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import gc

# --- ページ設定 ---
st.set_page_config(
    page_title="特徴量エンジニアリング支援アプリ",
    page_icon="🔧",
    layout="wide"
)

# --- 関数 ---
def reduce_mem_usage(df):
    """DataFrameのメモリ使用量を削減する"""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:3] == 'int':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        elif str(col_type)[:5] == 'float':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    st.toast(f'メモリ使用量を {start_mem:.2f} MB から {end_mem:.2f} MB に削減しました。', icon='🧠')
    return df

def load_csv(uploaded_file):
    """CSVを読み込み、メモリ最適化を行う"""
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        return reduce_mem_usage(df)
    except Exception as e:
        st.error(f"エラー: ファイルを読み込めませんでした。({e})")
        return None

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

def cramers_v(contingency_table):
    """クラメールの連関係数を計算する"""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    # --- ★★★ エラー修正箇所 ★★★ ---
    # この行の閉じ括弧が欠けていたのがエラーの原因でした
    if min((k_corr-1), (r_corr-1)) == 0:
        return 0
    return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))
    # --- ★★★★★★★★★★★★★★★★★ ---

def correlation_ratio(categories, measurements):
    """相関比を計算する"""
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.where(fcat == i)[0]]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0 or denominator == 0:
        return 0
    return np.sqrt(numerator/denominator)

# --- セッションステートの初期化 ---
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
    st.session_state.df_original = None
    st.session_state.generated_code = []

# --- メイン画面 ---
st.title("🔧 特徴量エンジニアリング支援アプリ")
st.write("CSVをアップロードし、サイドバーの機能を使って新しい特徴量を直感的に作成しましょう。")

# --- サイドバー ---
with st.sidebar:
    st.header("操作パネル")
    uploaded_file = st.file_uploader("1. CSVファイルをアップロード", type=["csv"])

    if uploaded_file:
        if st.button("データ読み込み実行", use_container_width=True):
            with st.spinner("データを読み込んで最適化しています..."):
                df = load_csv(uploaded_file)
                if df is not None:
                    st.session_state.df_original = df.copy()
                    st.session_state.df_processed = df.copy()
                    st.session_state.generated_code = []
                    st.success("ファイルの読み込みが完了しました。")
                    st.rerun()

    if st.session_state.df_processed is not None:
        df = st.session_state.df_processed
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()

        st.markdown("---")
        st.subheader("2. 特徴量を作成")
        
        # (各機能のロジックは変更なし - 中略)
        
        st.markdown("---")
        if st.button("🔄 変更をリセット"):
            st.session_state.df_processed = st.session_state.df_original.copy()
            st.session_state.generated_code = []
            gc.collect()
            st.rerun()

# --- メインエリアでの結果表示 ---
if st.session_state.df_processed is not None:
    df_display = st.session_state.df_processed
    st.subheader("✨ 加工後のデータフレーム")
    st.dataframe(df_display)

    # (出力機能は変更なし - 中略)
    
    st.markdown("---")
    with st.expander("📊 カラムごとの簡易分析"):
        pass
    
    st.markdown("---")
    st.header("🔗 相関分析")
    tab1, tab2, tab3 = st.tabs(["数値 vs 数値 (相関係数)", "カテゴリ vs カテゴリ (クラメールV)", "数値 vs カテゴリ (相関比)"])

    with tab1:
        st.subheader("相関係数ヒートマップ")
        numeric_cols_df = df_display.select_dtypes(include=np.number)
        if len(numeric_cols_df.columns) > 1:
            if st.button("ヒートマップを計算", key="corr_heatmap_btn"):
                with st.spinner("相関係数行列を計算中..."):
                    corr_matrix = numeric_cols_df.corr(numeric_only=True)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)
                    gc.collect()
        else:
            st.warning("相関を計算するには、少なくとも2つ以上の数値列が必要です。")
            
    with tab2:
        st.subheader("クラメールの連関係数")
        cat_cols_list = df_display.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(cat_cols_list) > 1:
            col1_select = st.selectbox("列 1 を選択", cat_cols_list, key="cramers_col1")
            col2_select = st.selectbox("列 2 を選択", cat_cols_list, index=min(1, len(cat_cols_list)-1), key="cramers_col2")

            if st.button("クラメールVを計算", key="cramers_run_btn"):
                if col1_select == col2_select:
                    st.warning("同じ列を選択しています。異なる列を選んでください。")
                else:
                    with st.spinner(f"「{col1_select}」と「{col2_select}」の連関係数を計算中..."):
                        contingency_table = pd.crosstab(df_display[col1_select], df_display[col2_select])
                        v = cramers_v(contingency_table)
                        st.metric(f"クラメールの連関係数 (V)", f"{v:.4f}")
                        
                        st.write("クロス集計表:")
                        st.dataframe(contingency_table)
        else:
            st.warning("クラメールVを計算するには、少なくとも2つ以上のカテゴリ列が必要です。")

    with tab3:
        pass

else:
    st.info("サイドバーからCSVファイルをアップロードし、「データ読み込み実行」ボタンを押して開始してください。")
