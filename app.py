import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px

# --- ページ設定 (変更なし) ---
st.set_page_config(
    page_title="特徴量エンジニアリング支援アプリ",
    page_icon="🔧",
    layout="wide"
)

# --- 関数 (変更なし) ---
@st.cache_data
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"エラー: ファイルを読み込めませんでした。({e})")
        return None

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- セッションステートの初期化 (変更なし) ---
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
    st.session_state.df_original = None
    st.session_state.generated_code = []

# --- メイン画面 (変更なし) ---
st.title("🔧 特徴量エンジニアリング支援アプリ")
st.write("CSVをアップロードし、サイドバーの機能を使って新しい特徴量を直感的に作成しましょう。")

# --- サイドバー ---
with st.sidebar:
    st.header("操作パネル")
    uploaded_file = st.file_uploader("1. CSVファイルをアップロード", type=["csv"])

    if uploaded_file is not None:
        if st.session_state.df_original is None:
            df = load_csv(uploaded_file)
            if df is not None:
                st.session_state.df_original = df.copy()
                st.session_state.df_processed = df.copy()
                st.success("ファイルを読み込みました。")

    if st.session_state.df_processed is not None:
        df = st.session_state.df_processed
        # --- ここからが変更点 ---
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include='object').columns.tolist()
        all_cols = df.columns.tolist()

        st.markdown("---")
        st.subheader("2. 特徴量を作成")

        # --- 数値列が存在する場合のみ、関連機能を表示 ---
        if numeric_cols:
            with st.expander("🔢 四則演算機能"):
                # (ヘルプ機能は省略)
                st.write("2つの数値列と定数で計算します。")
                col1 = st.selectbox("列1", numeric_cols, key="calc_col1")
                op = st.selectbox("演算子", ["+", "-", "*", "/"], key="calc_op")
                col2 = st.selectbox("列2", numeric_cols, key="calc_col2")
                const = st.number_input("定数（最後に加算/減算）", value=0.0, format="%.4f")
                new_col_name_calc = st.text_input("新しい列名", "calc_result", key="calc_new_col")
                if st.button("計算実行", key="calc_run"):
                    # (実行ロジックは変更なし)
                    pass

            with st.expander("📊 ビニング（カテゴリ化）機能"):
                # (ヘルプ機能は省略)
                st.write("連続値を任意の範囲で区切り、カテゴリ分けします。")
                col_to_bin = st.selectbox("対象の列", numeric_cols, key="bin_col")
                bins_str = st.text_input("区切り値 (カンマ区切り)", "0, 18, 40, 60, 100")
                labels_str = st.text_input("カテゴリ名 (カンマ区切り)", "Child, Young, Adult, Senior")
                new_col_name_bin = st.text_input("新しい列名", "binned_result", key="bin_new_col")
                if st.button("ビニング実行", key="bin_run"):
                    # (実行ロジックは変更なし)
                    pass
        else:
            # --- 数値列がない場合に警告を表示 ---
            st.warning("数値列がないため、「四則演算」と「ビニング」は使用できません。")


        with st.expander("🤔 条件分岐 (IF-TH-ELSE) 機能"):
             # (この機能は全カラム対象なので変更なし)
            pass

        # --- テキスト列が存在する場合のみ、関連機能を表示 ---
        if object_cols:
            with st.expander("✍️ テキスト処理 (正規表現で抽出)"):
                # (ヘルプ機能は省略)
                st.write("テキスト列から正規表現で特定のパターンを抽出します。")
                text_col = st.selectbox("対象の列", object_cols, key="re_col")
                regex_pattern = st.text_input("正規表現パターン", r'([A-Za-z]+)\.', key="re_pattern")
                new_col_name_re = st.text_input("新しい列名", "extracted_text", key="re_new_col")
                if st.button("抽出実行", key="re_run"):
                    # (実行ロジックは変更なし)
                    pass
        else:
            # --- テキスト列がない場合に警告を表示 ---
            st.warning("テキスト列がないため、「テキスト処理」は使用できません。")


        st.markdown("---")
        if st.button("🔄 変更をリセット"):
            st.session_state.df_processed = st.session_state.df_original.copy()
            st.session_state.generated_code = []
            st.rerun()

# --- メインエリアでの結果表示 (変更なし) ---
if st.session_state.df_processed is not None:
    # (省略)
    pass
else:
    st.info("サイドバーからCSVファイルをアップロードして開始してください。")
