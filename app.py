import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
# --- ここからが追加 ---
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
# --- ここまでが追加 ---

# --- ページ設定 (変更なし) ---
st.set_page_config(
    page_title="特徴量エンジニアリング支援アプリ",
    page_icon="🔧",
    layout="wide"
)

# --- 関数 ---
@st.cache_data
def load_csv(uploaded_file):
    """アップロードされたCSVをDataFrameとして読み込む"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"エラー: ファイルを読み込めませんでした。({e})")
        return None

def convert_df_to_csv(df):
    """DataFrameをCSV形式のバイトデータに変換する"""
    return df.to_csv(index=False).encode('utf-8-sig')

# --- ここからが追加 ---
def cramers_v(contingency_table):
    """クラメールの連関係数を計算する"""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))

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
    else:
        return np.sqrt(numerator/denominator)
# --- ここまでが追加 ---

# --- セッションステートの初期化 (変更なし) ---
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
    st.session_state.df_original = None
    st.session_state.generated_code = []

# --- メイン画面 (変更なし) ---
st.title("🔧 特徴量エンジニアリング支援アプリ")
st.write("CSVをアップロードし、サイドバーの機能を使って新しい特徴量を直感的に作成しましょう。")

# --- サイドバー (変更なし) ---
with st.sidebar:
    # (省略... 変更なし)
    pass

# --- メインエリアでの結果表示 ---
if st.session_state.df_processed is not None:
    df_display = st.session_state.df_processed
    st.subheader("✨ 加工後のデータフレーム")
    st.dataframe(df_display)

    st.markdown("---")
    with st.expander("📊 カラムごとの簡易分析"):
        # (省略... 変更なし)
        pass

    # --- ここからが追加機能 ---
    st.markdown("---")
    st.header("🔗 相関分析")
    st.write("特徴量同士、または特徴量とターゲットとの関係性を分析します。")

    tab1, tab2, tab3 = st.tabs([
        "数値 vs 数値 (相関係数)", 
        "カテゴリ vs カテゴリ (クラメールV)", 
        "数値 vs カテゴリ (相関比)"
    ])

    with tab1:
        st.subheader("相関係数ヒートマップ")
        numeric_cols = df_display.select_dtypes(include=np.number)
        if len(numeric_cols.columns) > 1:
            corr_matrix = numeric_cols.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("相関を計算するには、少なくとも2つ以上の数値列が必要です。")

    with tab2:
        st.subheader("クラメールの連関係数")
        object_cols = df_display.select_dtypes(include=['object', 'category']).columns
        if len(object_cols) > 1:
            cramers_v_matrix = pd.DataFrame(np.ones((len(object_cols), len(object_cols))), index=object_cols, columns=object_cols)
            for col1 in object_cols:
                for col2 in object_cols:
                    if col1 != col2:
                        contingency_table = pd.crosstab(df_display[col1], df_display[col2])
                        cramers_v_matrix.loc[col1, col2] = cramers_v(contingency_table)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cramers_v_matrix, annot=True, cmap='viridis', fmt='.2f', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("クラメールVを計算するには、少なくとも2つ以上のカテゴリ列が必要です。")

    with tab3:
        st.subheader("相関比")
        numeric_cols_list = df_display.select_dtypes(include=np.number).columns.tolist()
        object_cols_list = df_display.select_dtypes(include=['object', 'category']).columns.tolist()

        if numeric_cols_list and object_cols_list:
            selected_cat_col = st.selectbox("基準となるカテゴリ列を選択", object_cols_list)
            if selected_cat_col:
                corr_ratios = {}
                for num_col in numeric_cols_list:
                    ratio = correlation_ratio(df_display[selected_cat_col], df_display[num_col])
                    corr_ratios[num_col] = ratio
                
                corr_ratio_df = pd.DataFrame(list(corr_ratios.items()), columns=['数値列', '相関比']).sort_values(by='相関比', ascending=False)
                
                fig = px.bar(corr_ratio_df, x='相関比', y='数値列', orientation='h', title=f"カテゴリ列「{selected_cat_col}」と各数値列の相関比")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("相関比を計算するには、少なくとも1つずつの数値列とカテゴリ列が必要です。")

    # --- ここまでが追加機能 ---
    
    st.download_button(
       label="加工後のCSVをダウンロード",
       data=convert_df_to_csv(df_display),
       file_name='featured_data.csv',
       mime='text/csv',
    )
    
    if st.session_state.generated_code:
        st.subheader("🐍 生成されたPythonコード")
        st.info("以下のコードで、今回の操作を再現できます。")
        full_code = "\n\n".join(st.session_state.generated_code)
        st.code(full_code, language='python')

else:
    st.info("サイドバーからCSVファイルをアップロードして開始してください。")
