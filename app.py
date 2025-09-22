import streamlit as st
import pandas as pd
import numpy as np
import io

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

# --- サイドバー ---
with st.sidebar:
    st.header("操作パネル")
    uploaded_file = st.file_uploader("1. CSVファイルをアップロード", type=["csv"])

    # ファイルがアップロードされたらデータを読み込み、セッションステートに保存
    if uploaded_file is not None:
        if st.session_state.df_original is None:
            df = load_csv(uploaded_file)
            if df is not None:
                st.session_state.df_original = df.copy()
                st.session_state.df_processed = df.copy()
                st.success("ファイルを読み込みました。")

    # データが読み込まれている場合のみ操作を表示
    if st.session_state.df_processed is not None:
        df = st.session_state.df_processed
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        st.markdown("---")
        st.subheader("2. 特徴量を作成")

        # --- 各機能 ---
        with st.expander("🔢 四則演算機能"):
            # --- ヘルプ機能の追加 ---
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

        with st.expander("📊 ビニング（カテゴリ化）機能"):
            # --- ヘルプ機能の追加 ---
            with st.popover("使い方のヒント 💡"):
                st.markdown("""
                **目的**: 連続値の列（年齢など）をいくつかのカテゴリにグループ分けします。
                
                **具体例**: タイタニックデータで「年齢層」(`AgeGroup`) を作る場合
                - **対象の列**: `age`
                - **区切り値**: `0, 18, 60, 100` 
                - **カテゴリ名**: `Underage, Adult, Senior`
                
                **ポイント**: 「カテゴリ名」の数は「区切り値」の数より1つ少なくします。
                """)

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
                except Exception as e:
                    st.error(f"ビニングエラー: {e}")

        with st.expander("🤔 条件分岐 (IF-THEN-ELSE) 機能"):
            # --- ヘルプ機能の追加 ---
            with st.popover("使い方のヒント 💡"):
                st.markdown("""
                **目的**: ある条件を満たすかどうかで、新しい列に異なる値を設定します。（例: 0か1かのフラグ作成）
                
                **具体例**: タイタニックデータで「一人旅フラグ」(`IsAlone`) を作る場合
                - **IF: 対象の列**: `FamilySize`
                - **条件**: `==`
                - **値**: `1`
                - **THEN**: `1` (一人旅の場合)
                - **ELSE**: `0` (家族連れの場合)
                - **新しい列名**: `IsAlone`
                """)

            st.write("条件に合致する場合としない場合で値を設定します。")
            if_col = st.selectbox("IF: 対象の列", all_cols, key="if_col")
            if_op = st.selectbox("条件", ["==", "!=", ">", "<", ">=", "<="], key="if_op")
            if_val = st.text_input("値", "1", key="if_val")
            then_val = st.text_input("THEN: 設定する値", "1", key="if_then")
            else_val = st.text_input("ELSE: 設定する値", "0", key="if_else")
            new_col_name_if = st.text_input("新しい列名", "conditional_result", key="if_new_col")

            if st.button("条件分岐実行", key="if_run"):
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

        with st.expander("✍️ テキスト処理 (正規表現で抽出)"):
            # --- ヘルプ機能の追加 ---
            with st.popover("使い方のヒント 💡"):
                st.markdown(r"""
                **目的**: テキスト列から正規表現を使って、特定のパターンの文字列を抜き出します。
                
                **具体例**: タイタニックデータの`name`列から敬称(`Mr.`など)を抽出する場合
                - **対象の列**: `name`
                - **正規表現パターン**: `([A-Za-z]+)\.`
                - **新しい列名**: `Title`
                
                この正規表現は「アルファベットの単語の直後にピリオド(.)がある部分」を探します。
                """)
            
            st.write("テキスト列から正規表現で特定のパターンを抽出します。")
            text_col = st.selectbox("対象の列", df.select_dtypes(include='object').columns.tolist(), key="re_col")
            regex_pattern = st.text_input("正規表現パターン", r'([A-Za-z]+)\.', key="re_pattern")
            new_col_name_re = st.text_input("新しい列名", "extracted_text", key="re_new_col")

            if st.button("抽出実行", key="re_run"):
                try:
                    df[new_col_name_re] = df[text_col].str.extract(regex_pattern)
                    st.session_state.generated_code.append(f"df['{new_col_name_re}'] = df['{text_col}'].str.extract(r'{regex_pattern}')")
                    st.success(f"列 '{new_col_name_re}' を作成しました。")
                except Exception as e:
                    st.error(f"抽出エラー: {e}")

        st.markdown("---")
        if st.button("🔄 変更をリセット"):
            st.session_state.df_processed = st.session_state.df_original.copy()
            st.session_state.generated_code = []
            st.rerun()


# --- メインエリアでの結果表示 ---
if st.session_state.df_processed is not None:
    st.subheader("✨ 加工後のデータフレーム")
    st.dataframe(st.session_state.df_processed)

    st.download_button(
       label="加工後のCSVをダウンロード",
       data=convert_df_to_csv(st.session_state.df_processed),
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

