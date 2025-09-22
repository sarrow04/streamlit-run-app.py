import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px  # plotlyをインポート

# --- ページ設定 ---
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
    return df.to_csv(index=False).encode('utf-8')

# --- セッションステートの初期化 ---
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
    st.session_state.df_original = None
    st.session_state.generated_code = []

# --- メイン画面 ---
st.title("🔧 特徴量エンジニアリング支援アプリ")
st.write("CSVをアップロードし、サイドバーの機能を使って新しい特徴量を直感的に作成しましょう。")

# --- サイドバー (変更なし) ---
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
        all_cols = df.columns.tolist()

        st.markdown("---")
        st.subheader("2. 特徴量を作成")

        # --- 各機能 (変更なし) ---
        with st.expander("🔢 四則演算機能"):
            with st.popover("使い方のヒント 💡"):
                st.markdown("""
                **目的**: 2つの数値列と定数を使って新しい列を計算します。
                
                **具体例**: タイタニックデータで「家族の人数」(`FamilySize`) を作る場合
                - **列1**: `sibsp`
                - **演算子**: `+`
                - **列2**: `parch`
                - **定数**: `1` (乗客本人分)
                - **新しい列名**: `FamilySize`
                """)
            
            st.write("2つの数値列と定数で計算します。")
            col1 = st.selectbox("列1", numeric_cols, key="calc_col1")
            op = st.selectbox("演算子", ["+", "-", "*", "/"], key="calc_op")
            col2 = st.selectbox("列2", numeric_cols, key="calc_col2")
            const = st.number_input("定数（最後に加算/減算）", value=0.0, format="%.4f")
            new_col_name_calc = st.text_input("新しい列名", "calc_result", key="calc_new_col")

            if st.button("計算実行", key="calc_run"):
                try:
                    expr = f"df['{col1}'] {op} df['{col2}'] + {const}"
                    df[new_col_name_calc] = pd.eval(expr)
                    st.session_state.generated_code.append(f"df['{new_col_name_calc}'] = {expr}")
                    st.success(f"列 '{new_col_name_calc}' を作成しました。")
                except Exception as e:
                    st.error(f"計算エラー: {e}")

        # ... (他の機能のコードは変更がないため省略) ...
        with st.expander("📊 ビニング（カテゴリ化）機能"):
            # (省略)
            pass
        with st.expander("🤔 条件分岐 (IF-THEN-ELSE) 機能"):
            # (省略)
            pass
        with st.expander("✍️ テキスト処理 (正規表現で抽出)"):
            # (省略)
            pass
            
        st.markdown("---")
        if st.button("🔄 変更をリセット"):
            st.session_state.df_processed = st.session_state.df_original.copy()
            st.session_state.generated_code = []
            st.rerun()


# --- メインエリアでの結果表示 ---
if st.session_state.df_processed is not None:
    df_display = st.session_state.df_processed
    st.subheader("✨ 加工後のデータフレーム")
    st.dataframe(df_display)

    # --- ここからが追加した機能 ---
    st.markdown("---")
    with st.expander("📊 カラムごとの簡易分析"):
        
        # 分析したいカラムをユーザーに選択させる
        selected_column = st.selectbox(
            "分析したいカラムを選択してください",
            df_display.columns
        )

        if selected_column:
            # 1. 基本統計量の表示
            st.subheader(f"基本統計量: `{selected_column}`")
            st.dataframe(df_display[selected_column].describe())

            # 2. グラフの表示
            st.subheader(f"分布の可視化: `{selected_column}`")
            
            # カラムが数値型かどうかで処理を分岐
            if pd.api.types.is_numeric_dtype(df_display[selected_column]):
                # 数値型ならヒストグラムを表示
                fig = px.histogram(df_display, x=selected_column, title=f'`{selected_column}`のヒストグラム')
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 数値型でなければ、度数分布の棒グラフを表示
                st.info("このカラムは数値ではないため、各カテゴリの出現回数を表示します。")
                value_counts = df_display[selected_column].value_counts().reset_index()
                value_counts.columns = [selected_column, 'count']
                fig = px.bar(value_counts, x=selected_column, y='count', title=f'`{selected_column}`の度数分布')
                st.plotly_chart(fig, use_container_width=True)

    # --- ここまでが追加した機能 ---

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
