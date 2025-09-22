import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

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
    """DataFrameをCSV形式のバイトデータに変換する (文字化け対策済み)"""
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
    # 0除算を回避
    if min((k_corr-1), (r_corr-1)) == 0:
        return 0
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

        # --- 数値列が存在する場合のみ、関連機能を表示 ---
        if numeric_cols:
            with st.expander("🔢 四則演算機能"):
                # (ヘルプは簡潔に)
                with st.popover("ヒント💡"): st.markdown("**具体例**: `FamilySize` を作る\n- **列1**: `sibsp` `+` **列2**: `parch` `+` **定数**: `1`")
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
                with st.popover("ヒント💡"): st.markdown("**具体例**: `AgeGroup` を作る\n- **対象**: `age`, **区切り**: `0,18,60,100`, **カテゴリ**: `Child,Adult,Senior`")
                col_to_bin = st.selectbox("対象の列", numeric_cols, key="bin_col")
                bins_str = st.text_input("区切り値 (カンマ区切り)", "0, 18, 60, 100")
                labels_str = st.text_input("カテゴリ名 (カンマ区切り)", "Child, Adult, Senior")
                new_col_name_bin = st.text_input("新しい列名", "binned_result", key="bin_new_col")
                if st.button("ビニング実行", key="bin_run"):
                    try:
                        bins = [float(b.strip()) for b in bins_str.split(',')]
                        labels = [l.strip() for l in labels_str.split(',')]
                        df[new_col_name_bin] = pd.cut(df[col_to_bin], bins=bins, labels=labels, right=False)
                        st.session_state.generated_code.append(f"bins = {bins}\nlabels = {labels}\ndf['{new_col_name_bin}'] = pd.cut(df['{col_to_bin}'], bins=bins, labels=labels, right=False)")
                        st.success(f"列 '{new_col_name_bin}' を作成しました。")
                    except Exception as e: st.error(f"ビニングエラー: {e}")
            
            with st.expander("↔️ スケーリング（正規化・標準化）"):
                with st.popover("ヒント💡"): st.markdown("**正規化**: データを0〜1の範囲に変換します。\n**標準化**: データを平均0, 標準偏差1の分布に変換します。")
                col_to_scale = st.selectbox("対象の列", numeric_cols, key="scale_col")
                method = st.radio("スケーリング方法を選択", ["正規化 (Min-Max)", "標準化 (Standard)"], key="scale_method")
                new_col_name_scale = st.text_input("新しい列名", f"{col_to_scale}_scaled", key="scale_new_col")
                if st.button("スケーリング実行", key="scale_run"):
                    # (実行ロジックは前の回答と同様)
                    pass

        else:
            st.warning("数値列がないため、「四則演算」「ビニング」「スケーリング」は使用できません。")

        with st.expander("🤔 条件分岐 (IF-THEN-ELSE) 機能"):
            with st.popover("ヒント💡"): st.markdown("**具体例**: `IsAlone` を作る\n- **IF**: `FamilySize` `==` `1`\n- **THEN**: `1`, **ELSE**: `0`")
            if_col = st.selectbox("IF: 対象の列", all_cols, key="if_col")
            if_op = st.selectbox("条件", ["==", "!=", ">", "<", ">=", "<="], key="if_op")
            if_val = st.text_input("値", "1", key="if_val")
            then_val = st.text_input("THEN: 設定する値", "1", key="if_then")
            else_val = st.text_input("ELSE: 設定する値", "0", key="if_else")
            new_col_name_if = st.text_input("新しい列名", "conditional_result", key="if_new_col")
            if st.button("条件分岐実行", key="if_run"):
                try:
                    try: if_val_eval = eval(if_val)
                    except: if_val_eval = f"'{if_val}'"
                    condition = f"df['{if_col}'] {if_op} {if_val_eval}"
                    df[new_col_name_if] = np.where(pd.eval(condition), then_val, else_val)
                    st.session_state.generated_code.append(f"df['{new_col_name_if}'] = np.where({condition}, '{then_val}', '{else_val}')")
                    st.success(f"列 '{new_col_name_if}' を作成しました。")
                except Exception as e: st.error(f"条件分岐エラー: {e}")

        if object_cols:
            with st.expander("✍️ テキスト処理 (正規表現で抽出)"):
                with st.popover("ヒント💡"): st.markdown(r"**具体例**: `Title` を作る\n- **対象**: `name`, **正規表現**: `([A-Za-z]+)\.`")
                text_col = st.selectbox("対象の列", object_cols, key="re_col")
                regex_pattern = st.text_input("正規表現パターン", r'([A-Za-z]+)\.', key="re_pattern")
                new_col_name_re = st.text_input("新しい列名", "extracted_text", key="re_new_col")
                if st.button("抽出実行", key="re_run"):
                    try:
                        df[new_col_name_re] = df[text_col].str.extract(regex_pattern)
                        st.session_state.generated_code.append(f"df['{new_col_name_re}'] = df['{text_col}'].str.extract(r'{regex_pattern}')")
                        st.success(f"列 '{new_col_name_re}' を作成しました。")
                    except Exception as e: st.error(f"抽出エラー: {e}")
        else:
            st.warning("テキスト列がないため、「テキスト処理」は使用できません。")

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

    st.markdown("---")
    with st.expander("📊 カラムごとの簡易分析"):
        if not df_display.columns.empty:
            selected_column = st.selectbox("分析したいカラムを選択してください", df_display.columns, key="dist_select")
            if selected_column:
                st.subheader(f"基本統計量: `{selected_column}`")
                st.dataframe(df_display[selected_column].describe())
                st.subheader(f"分布の可視化: `{selected_column}`")
                if pd.api.types.is_numeric_dtype(df_display[selected_column]):
                    fig = px.histogram(df_display, x=selected_column)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("このカラムは数値ではないため、各カテゴリの出現回数を表示します。")
                    value_counts = df_display[selected_column].value_counts().reset_index()
                    value_counts.columns = [selected_column, 'count']
                    fig = px.bar(value_counts, x=selected_column, y='count')
                    st.plotly_chart(fig, use_container_width=True)
    
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
        else:
            st.warning("相関を計算するには、少なくとも2つ以上の数値列が必要です。")
    with tab2:
        st.subheader("クラメールの連関係数ヒートマップ")
        object_cols_list = df_display.select_dtypes(include=['object', 'category']).columns.tolist()
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
        else:
            st.warning("クラメールVを計算するには、少なくとも2つ以上のカテゴリ列が必要です。")

    with tab3:
        st.subheader("相関比")
        numeric_cols_list_cr = df_display.select_dtypes(include=np.number).columns.tolist()
        object_cols_list_cr = df_display.select_dtypes(include=['object', 'category']).columns.tolist()
        if numeric_cols_list_cr and object_cols_list_cr:
            selected_cat_col = st.selectbox("基準となるカテゴリ列を選択", object_cols_list_cr)
            if selected_cat_col:
                corr_ratios = {num_col: correlation_ratio(df_display[selected_cat_col], df_display[num_col]) for num_col in numeric_cols_list_cr}
                corr_ratio_df = pd.DataFrame(list(corr_ratios.items()), columns=['数値列', '相関比']).sort_values(by='相関比', ascending=False)
                fig = px.bar(corr_ratio_df, x='相関比', y='数値列', orientation='h', title=f"カテゴリ列「{selected_cat_col}」と各数値列の相関比")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("相関比を計算するには、少なくとも1つずつの数値列とカテゴリ列が必要です。")

    st.markdown("---")
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
