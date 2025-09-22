import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px

# --- ページ設定 ---
st.set_page_config(
    page_title="特徴量エンジニアリング支援アプリ",
    page_icon="🔧",
    layout="wide"
)

# --- 関数 ---
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
        if st.session_state.df_original is None:
            df = load_csv(uploaded_file)
            if df is not None:
                st.session_state.df_original = df.copy()
                st.session_state.df_processed = df.copy()
                st.success("ファイルを読み込みました。")

    if st.session_state.df_processed is not None:
        df = st.session_state.df_processed
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include='object').columns.tolist()
        all_cols = df.columns.tolist()

        st.markdown("---")
        st.subheader("2. 特徴量を作成")

        if numeric_cols:
            with st.expander("🔢 四則演算機能"):
                # (ヘルプは省略)
                st.write("2つの数値列と定数で計算します。")
                col1 = st.selectbox("列1", numeric_cols, key="calc_col1")
                op = st.selectbox("演算子", ["+", "-", "*", "/"], key="calc_op")
                col2 = st.selectbox("列2", numeric_cols, key="calc_col2")
                const = st.number_input("定数", value=0.0, format="%.4f")
                new_col_name_calc = st.text_input("新しい列名", "calc_result", key="calc_new_col")
                if st.button("計算実行", key="calc_run"):
                    try:
                        expr = f"df['{col1}'] {op} df['{col2}'] + {const}"
                        df[new_col_name_calc] = pd.eval(expr)
                        st.session_state.generated_code.append(f"df['{new_col_name_calc}'] = {expr}")
                        st.success(f"列 '{new_col_name_calc}' を作成しました。")
                    except Exception as e: st.error(f"計算エラー: {e}")

            with st.expander("📊 ビニング（カテゴリ化）機能"):
                # (ヘルプは省略)
                st.write("連続値を任意の範囲で区切り、カテゴリ分けします。")
                col_to_bin = st.selectbox("対象の列", numeric_cols, key="bin_col")
                bins_str = st.text_input("区切り値 (カンマ区切り)", "0, 18, 40, 60, 100")
                labels_str = st.text_input("カテゴリ名 (カンマ区切り)", "Child, Young, Adult, Senior")
                new_col_name_bin = st.text_input("新しい列名", "binned_result", key="bin_new_col")
                if st.button("ビニング実行", key="bin_run"):
                    try:
                        bins = [float(b.strip()) for b in bins_str.split(',')]
                        labels = [l.strip() for l in labels_str.split(',')]
                        df[new_col_name_bin] = pd.cut(df[col_to_bin], bins=bins, labels=labels, right=False)
                        st.session_state.generated_code.append(f"bins = {bins}\nlabels = {labels}\ndf['{new_col_name_bin}'] = pd.cut(df['{col_to_bin}'], bins=bins, labels=labels, right=False)")
                        st.success(f"列 '{new_col_name_bin}' を作成しました。")
                    except Exception as e: st.error(f"ビニングエラー: {e}")
        else:
            st.warning("数値列がないため、「四則演算」と「ビニング」は使用できません。")

        with st.expander("🤔 条件分岐 (IF-THEN-ELSE) 機能"):
            # (ヘルプは省略)
            st.write("条件に合致する場合としない場合で値を設定します。")
            if_col = st.selectbox("IF: 対象の列", all_cols, key="if_col")
            if_op = st.selectbox("条件", ["==", "!=", ">", "<", ">=", "<="], key="if_op")
            if_val = st.text_input("値", "1", key="if_val")
            then_val = st.text_input("THEN: 設定する値", "1", key="if_then")
            else_val = st.text_input("ELSE: 設定する値", "0", key="if_else")
            new_col_name_if = st.text_input("新しい列名", "conditional_result", key="if_new_col")
            if st.button("条件分岐実行", key="if_run"):
                # --- ここが修正されたブロックです ---
                try:
                    try:
                        if_val_eval = eval(if_val)
                    except:
                        if_val_eval = f"'{if_val}'"
                    
                    condition = f"df['{if_col}'] {if_op} {if_val_eval}"
                    df[new_col_name_if] = np.where(pd.eval(condition), then_val, else_val)
                    st.session_state.generated_code.append(f"df['{new_col_name_if}'] = np.where({condition}, '{then_val}', '{else_val}')")
                    st.success(f"列 '{new_col_name_if}' を作成しました。")
                except Exception as e: 
                    st.error(f"条件分岐エラー: {e}")

        if object_cols:
            with st.
