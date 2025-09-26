import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import gc # --- 改善点: ガーベジコレクションをインポート ---

# --- ページ設定 ---
st.set_page_config(
    page_title="特徴量エンジニアリング支援アプリ",
    page_icon="🔧",
    layout="wide"
)

# --- 関数 ---

### --- 改善点1: DataFrameのメモリ使用量を削減する関数を追加 --- ###
def reduce_mem_usage(df, verbose=True):
    """DataFrameの各列のデータ型を最適なものに変換し、メモリ使用量を削減する"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        st.toast(f'メモリ使用量を {start_mem:.2f} MB から {end_mem:.2f} MB に削減しました ({-100 * (start_mem - end_mem) / start_mem:.1f} % 削減)。', icon='🧠')
    return df

@st.cache_data
def load_and_optimize_csv(uploaded_file):
    """アップロードされたCSVを読み込み、メモリ最適化を行う"""
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        # 読み込んだ直後にメモリ削減を実行
        df_optimized = reduce_mem_usage(df)
        return df_optimized
    except Exception as e:
        st.error(f"エラー: ファイルを読み込めませんでした。({e})")
        return None

def convert_df_to_csv(df):
    """DataFrameをCSV形式のバイトデータに変換する (文字化け対策済み)"""
    return df.to_csv(index=False).encode('utf-8-sig')

# (cramers_v, correlation_ratio 関数は変更なし)
def cramers_v(contingency_table):
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    if min((k_corr-1), (r_corr-1)) == 0:
        return 0
    return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))

def correlation_ratio(categories, measurements):
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
    else:
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

    if uploaded_file is not None:
        ### --- 改善点2: メモリ最適化された関数を呼び出す --- ###
        df = load_and_optimize_csv(uploaded_file)
        if df is not None and (st.session_state.df_original is None or not df.equals(st.session_state.df_original)):
            st.session_state.df_original = df.copy()
            st.session_state.df_processed = df.copy()
            st.session_state.generated_code = []
            st.success("ファイルを読み込み、メモリを最適化しました。")

    if st.session_state.df_processed is not None:
        df = st.session_state.df_processed
        # (サイドバーのUI部分は変更なし)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()

        st.markdown("---")
        st.subheader("2. 特徴量を作成")
        
        # (各機能のロジックは変更なし)
        # ... 四則演算 ...
        if numeric_cols:
            with st.expander("🔢 四則演算機能"):
                # (中略)
                pass
        # ... ビニング ...
            with st.expander("📊 ビニング（カテゴリ化）機能"):
                # (中略)
                pass
        # ... スケーリング ...
            with st.expander("↔️ スケーリング（正規化・標準化）"):
                # (中略)
                pass
        # ... 条件分岐 ...
        with st.expander("🤔 条件分岐 (IF-THEN-ELSE) 機能"):
            # (中略)
            pass
        # ... テキスト処理 ...
        if object_cols:
            with st.expander("✍️ テキスト処理 (正規表現で抽出)"):
                # (中略)
                pass
        
        st.markdown("---")
        if st.button("🔄 変更をリセット"):
            st.session_state.df_processed = st.session_state.df_original.copy()
            st.session_state.generated_code = []
            gc.collect() # ガーベジコレクションを呼ぶ
            st.rerun()

# --- メインエリアでの結果表示 ---
if st.session_state.df_processed is not None:
    df_display = st.session_state.df_processed
    st.subheader("✨ 加工後のデータフレーム")
    st.dataframe(df_display)

    # (出力機能、簡易分析機能は変更なし)
    # ...

    st.markdown("---")
    st.header("🔗 相関分析")
    tab1, tab2, tab3 = st.tabs(["数値 vs 数値 (相関係数)", "カテゴリ vs カテゴリ (クラメールV)", "数値 vs カテゴリ (相関比)"])

    with tab1:
        st.subheader("相関係数ヒートマップ")
        numeric_cols_df = df_display.select_dtypes(include=np.number)
        if len(numeric_cols_df.columns) > 1:
            corr_matrix = numeric_cols_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
            ### --- 改善点3: プロット後にメモリを解放 --- ###
            plt.close(fig)
            del corr_matrix
            gc.collect()
        else:
            st.warning("相関を計算するには、少なくとも2つ以上の数値列が必要です。")

    with tab2:
        st.subheader("クラメールの連関係数ヒートマップ")
        # .astype('category') を追加してメモリ効率を上げる
        object_cols_df = df_display.select_dtypes(include=['object', 'category'])
        object_cols_list = object_cols_df.columns.tolist()

        if len(object_cols_list) > 1:
            cramers_v_matrix = pd.DataFrame(np.ones((len(object_cols_list), len(object_cols_list))), index=object_cols_list, columns=object_cols_list)
            for col1 in object_cols_list:
                for col2 in object_cols_list:
                    if col1 != col2:
                        contingency_table = pd.crosstab(df_display[col1], df_display[col2])
                        cramers_v_matrix.loc[col1, col2] = cramers_v(contingency_table)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cramers_v_matrix, annot=True, cmap='viridis', fmt='.2f', ax=ax)
            st.pyplot(fig)
            ### --- 改善点3: プロット後にメモリを解放 --- ###
            plt.close(fig)
            del cramers_v_matrix
            gc.collect()
        else:
            st.warning("クラメールVを計算するには、少なくとも2つ以上のカテゴリ列が必要です。")
    
    # (相関比タブは変更なし)
    with tab3:
        # (中略)
        pass

else:
    st.info("サイドバーからCSVファイルをアップロードして開始してください。")

# (サイドバーやメインエリアの機能詳細部分は省略していますが、
# 上記の変更点をあなたのコードに組み込んでください。)
