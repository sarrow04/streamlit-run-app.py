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
# (reduce_mem_usage, load_csv, convert_df_to_csv, cramers_v, correlation_ratio は変更なし)
def reduce_mem_usage(df):
    """DataFrameのメモリ使用量を削減する"""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if pd.api.types.is_numeric_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
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
        if len(cat_measures) == 0: continue
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
    st.session_state.freq_col_selected = None

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
                    st.session_state.freq_col_selected = None
                    st.success("ファイルの読み込みが完了しました。")
                    st.rerun()

    if st.session_state.df_processed is not None:
        df = st.session_state.df_processed
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()

        st.markdown("---")
        st.subheader("2. 特徴量を作成・分析")

        if numeric_cols:
            with st.expander("🔢 四則演算機能"):
                # ... (変更なし)
                with st.popover("ヒント💡"): st.markdown("**例**: `sibsp` + `parch` + `1` => `FamilySize`")
                col1 = st.selectbox("列1", numeric_cols, key="calc_col1")
                op = st.selectbox("演算子", ["+", "-", "*", "/"], key="calc_op")
                col2 = st.selectbox("列2", numeric_cols, key="calc_col2", index=min(1, len(numeric_cols)-1))
                const = st.number_input("定数", value=0.0, format="%.4f")
                new_col_name_calc = st.text_input("新しい列名", "calc_result", key="calc_new_col")
                if st.button("計算実行", key="calc_run"):
                    try:
                        expr = f"df['{col1}'] {op} df['{col2}'] + {const}"
                        df[new_col_name_calc] = pd.eval(expr)
                        st.session_state.generated_code.append(f"df['{new_col_name_calc}'] = df['{col1}'] {op} df['{col2}'] + {const}")
                        st.success(f"列 '{new_col_name_calc}' を作成しました。")
                        st.rerun()
                    except Exception as e: st.error(f"計算エラー: {e}")
            with st.expander("📊 ビニング（カテゴリ化）機能"):
                # ... (変更なし)
                with st.popover("ヒント💡"): st.markdown("**例**: `age` を `0,18,60,100` で区切り `Child,Adult,Senior` に")
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
                        st.rerun()
                    except Exception as e: st.error(f"ビニングエラー: {e}")
            with st.expander("↔️ スケーリング"):
                # ... (変更なし)
                with st.popover("ヒント💡"): st.markdown("**正規化**: 0〜1の範囲に変換\n**標準化**: 平均0, 標準偏差1に変換")
                col_to_scale = st.selectbox("対象の列", numeric_cols, key="scale_col")
                method = st.radio("方法", ["正規化 (Min-Max)", "標準化 (Standard)"], key="scale_method")
                new_col_name_scale = st.text_input("新しい列名", f"{col_to_scale}_scaled", key="scale_new_col")
                if st.button("スケーリング実行", key="scale_run"):
                    try:
                        col_data = df[[col_to_scale]]
                        if method == "正規化 (Min-Max)":
                            scaler, code_line = MinMaxScaler(), f"from sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler()\ndf['{new_col_name_scale}'] = scaler.fit_transform(df[['{col_to_scale}']])"
                        else:
                            scaler, code_line = StandardScaler(), f"from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ndf['{new_col_name_scale}'] = scaler.fit_transform(df[['{col_to_scale}']])"
                        df[new_col_name_scale] = scaler.fit_transform(col_data)
                        st.session_state.generated_code.append(code_line)
                        st.success(f"列 '{new_col_name_scale}' を作成しました。")
                        st.rerun()
                    except Exception as e: st.error(f"スケーリングエラー: {e}")
        else:
            st.warning("数値列がないため、一部機能は使用できません。")

        with st.expander("🤔 条件分岐 (IF-THEN-ELSE) 機能"):
            with st.popover("ヒント💡"): st.markdown("**例**: IF `FamilySize` `==` `1` THEN `1` ELSE `0` => `IsAlone`")
            if_col = st.selectbox("IF: 対象の列", all_cols, key="if_col")
            if_op = st.selectbox("条件", ["==", "!=", ">", "<", ">=", "<=", "in", "not in", "str.contains"], key="if_op")
            if_val_input = st.text_input("値", "1", key="if_val")
            then_val = st.text_input("THEN: 設定する値", "1", key="if_then")
            else_val = st.text_input("ELSE: 設定する値", "0", key="if_else")
            new_col_name_if = st.text_input("新しい列名", "conditional_result", key="if_new_col")
            
            if st.button("条件分岐実行", key="if_run"):
                try:
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # --- ここからが修正箇所 ---
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    target_series = df[if_col]
                    
                    # 比較する値の型を、対象の列の型に合わせて自動で変換
                    if pd.api.types.is_numeric_dtype(target_series.dtype) and if_op not in ["in", "not in"]:
                        try:
                            if_val = float(if_val_input) # 数値に変換
                        except ValueError:
                            st.error(f"エラー: 「{if_col}」は数値列です。比較する値には数値を入力してください。")
                            st.stop()
                    else:
                        if_val = if_val_input # 文字列として扱う

                    # 条件式を動的に生成
                    if if_op == "==": condition = (target_series == if_val)
                    elif if_op == "!=": condition = (target_series != if_val)
                    elif if_op == ">": condition = (target_series > if_val)
                    elif if_op == "<": condition = (target_series < if_val)
                    elif if_op == ">=": condition = (target_series >= if_val)
                    elif if_op == "<=": condition = (target_series <= if_val)
                    elif if_op in ["in", "not in"]:
                        # in/not in の場合は、カンマ区切りのリストとして解釈
                        val_list = [v.strip() for v in if_val.split(',')]
                        condition = target_series.isin(val_list)
                        if if_op == "not in":
                            condition = ~condition # 条件を反転
                    elif if_op == "str.contains":
                        condition = target_series.str.contains(if_val, na=False)
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    
                    df[new_col_name_if] = np.where(condition, then_val, else_val)
                    
                    # 生成コードも改善
                    generated_code_line = f"df['{new_col_name_if}'] = np.where(df['{if_col}'] {if_op} {repr(if_val)}, '{then_val}', '{else_val}')"
                    if if_op == "str.contains":
                        generated_code_line = f"df['{new_col_name_if}'] = np.where(df['{if_col}'].str.contains({repr(if_val)}, na=False), '{then_val}', '{else_val}')"
                    elif if_op in ["in", "not in"]:
                        val_list = [v.strip() for v in if_val.split(',')]
                        op_str = ".isin" if if_op == "in" else ".isin" # isinでコード生成し、コメントでnotを補足
                        prefix = "" if if_op == "in" else "~"
                        generated_code_line = f"df['{new_col_name_if}'] = np.where({prefix}df['{if_col}']{op_str}({val_list}), '{then_val}', '{else_val}')"

                    st.session_state.generated_code.append(generated_code_line)
                    st.success(f"列 '{new_col_name_if}' を作成しました。")
                    st.rerun()
                except Exception as e:
                    st.error(f"条件分岐エラー: {e}")

        if object_cols:
            with st.expander("✍️ テキスト処理 (正規表現)"):
                # ... (変更なし)
                with st.popover("ヒント💡"): st.markdown(r"**例**: `name` から `([A-Za-z]+)\.` で敬称を抽出 => `Title`")
                text_col = st.selectbox("対象の列", object_cols, key="re_col")
                regex_pattern = st.text_input("正規表現パターン", r'([A-Za-z]+)\.', key="re_pattern")
                new_col_name_re = st.text_input("新しい列名", "extracted_text", key="re_new_col")
                if st.button("抽出実行", key="re_run"):
                    try:
                        df[new_col_name_re] = df[text_col].str.extract(regex_pattern)
                        st.session_state.generated_code.append(f"df['{new_col_name_re}'] = df['{text_col}'].str.extract(r'{regex_pattern}')")
                        st.success(f"列 '{new_col_name_re}' を作成しました。")
                        st.rerun()
                    except Exception as e: st.error(f"抽出エラー: {e}")
            with st.expander("📊 文字の出現数を確認 (頻度分析)"):
                # ... (変更なし)
                with st.popover("ヒント💡"): st.markdown("カテゴリカルな列（文字の列）で、どの値が何回出現するかを確認します。")
                freq_col = st.selectbox("対象の列", object_cols, key="freq_col")
                st.session_state.freq_col_selected = freq_col
        else:
            st.warning("テキスト列がないため、「テキスト処理」関連機能は使用できません。")
        
        st.markdown("---")
        if st.button("🔄 変更をリセット"):
            st.session_state.df_processed = st.session_state.df_original.copy()
            st.session_state.generated_code = []
            st.session_state.freq_col_selected = None
            gc.collect()
            st.rerun()

# --- メインエリアでの結果表示 ---
if st.session_state.df_processed is not None:
    df_display = st.session_state.df_processed
    st.subheader("✨ 加工後のデータフレーム")
    st.dataframe(df_display)

    st.markdown("---")
    st.header("📤 出力")
    # ... (ダウンロードとコード表示部分は変更なし)
    st.download_button(
       label="加工後のCSVをダウンロード",
       data=convert_df_to_csv(df_display),
       file_name='featured_data.csv',
       mime='text/csv',
       use_container_width=True
    )
    if st.session_state.generated_code:
        with st.expander("🐍 生成されたPythonコードを見る"):
            st.info("以下のコードで、今回の操作を再現できます。")
            full_code = "\n\n".join(st.session_state.generated_code)
            st.code(full_code, language='python')

    if 'freq_col_selected' in st.session_state and st.session_state.freq_col_selected:
        # ... (変更なし)
        selected_freq_col = st.session_state.freq_col_selected
        st.markdown("---")
        st.header(f"🔍 「{selected_freq_col}」の出現数分析")

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write(f"**出現回数 トップ20**")
            value_counts_df = df_display[selected_freq_col].value_counts().reset_index()
            value_counts_df.columns = [selected_freq_col, '出現回数']
            st.dataframe(value_counts_df.head(20))

        with col2:
            st.write(f"**グラフ表示 トップ20**")
            top20_df = value_counts_df.head(20)
            if not top20_df.empty:
                fig = px.bar(top20_df, 
                             x='出現回数', 
                             y=selected_freq_col, 
                             orientation='h', 
                             title=f'「{selected_freq_col}」の出現回数トップ20')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("この列にはデータがありません。")

    st.markdown("---")
    with st.expander("📊 カラムごとの簡易分析"):
        # ... (変更なし)
        if not df_display.columns.empty:
            selected_column = st.selectbox("分析したいカラムを選択", df_display.columns)
            if selected_column:
                st.write(f"**基本統計量: `{selected_column}`**")
                st.dataframe(df_display[selected_column].describe())
                st.write(f"**分布の可視化: `{selected_column}`**")
                if pd.api.types.is_numeric_dtype(df_display[selected_column]):
                    fig = px.histogram(df_display, x=selected_column)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("カテゴリ変数のため、各値の出現回数を表示します。")
                    fig = px.bar(df_display[selected_column].value_counts())
                    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.header("🔗 相関分析")
    # ... (相関分析部分は変更なし)
    tab1, tab2, tab3 = st.tabs(["数値 vs 数値", "カテゴリ vs カテゴリ", "数値 vs カテゴリ"])
    with tab1:
        st.subheader("相関係数ヒートマップ")
        numeric_cols_df = df_display.select_dtypes(include=np.number)
        if len(numeric_cols_df.columns) > 1:
            if st.button("ヒートマップを計算", key="corr_heatmap_btn"):
                with st.spinner("計算中..."):
                    corr_matrix = numeric_cols_df.corr(numeric_only=True)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                    st.pyplot(fig)
                    plt.close(fig); gc.collect()
        else: st.warning("少なくとも2つ以上の数値列が必要です。")
    with tab2:
        st.subheader("クラメールの連関係数")
        cat_cols_list = df_display.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(cat_cols_list) > 1:
            col1, col2 = st.columns(2)
            col1_select = col1.selectbox("列 1", cat_cols_list, key="cramers_col1")
            col2_select = col2.selectbox("列 2", cat_cols_list, index=min(1, len(cat_cols_list)-1), key="cramers_col2")
            if st.button("クラメールVを計算", key="cramers_run_btn"):
                if col1_select == col2_select:
                    st.warning("異なる列を選んでください。")
                else:
                    with st.spinner("計算中..."):
                        contingency_table = pd.crosstab(df_display[col1_select], df_display[col2_select])
                        v = cramers_v(contingency_table)
                        st.metric(f"クラメールの連関係数 (V)", f"{v:.4f}")
                        st.dataframe(contingency_table)
        else: st.warning("少なくとも2つ以上のカテゴリ列が必要です。")
    with tab3:
        st.subheader("相関比")
        numeric_cols_cr = df_display.select_dtypes(include=np.number).columns.tolist()
        cat_cols_cr = df_display.select_dtypes(include=['object', 'category']).columns.tolist()
        if numeric_cols_cr and cat_cols_cr:
            selected_cat_col = st.selectbox("基準となるカテゴリ列", cat_cols_cr)
            if st.button("相関比を計算", key="corr_ratio_btn"):
                with st.spinner("計算中..."):
                    corr_ratios = {num_col: correlation_ratio(df_display[selected_cat_col], df_display[num_col]) for num_col in numeric_cols_cr}
                    corr_ratio_df = pd.DataFrame(list(corr_ratios.items()), columns=['数値列', '相関比']).sort_values('相関比', ascending=False)
                    fig = px.bar(corr_ratio_df, x='相関比', y='数値列', orientation='h', title=f"「{selected_cat_col}」と各数値列の相関比")
                    st.plotly_chart(fig, use_container_width=True)
        else: st.warning("少なくとも1つずつの数値列とカテゴリ列が必要です。")
else:
    st.info("サイドバーからCSVファイルをアップロードし、「データ読み込み実行」ボタンを押して開始してください。")