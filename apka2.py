import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from groq import Groq
import os
from scipy.stats import chi2_contingency


def run():
    #-----Konfiguracia-------
    st.title("Data App")
    st.markdown("Тest vs Control, p-value, AI-висновки ")
    
    
    @st.cache_data
    def load_data(path: str):
        return pd.read_csv(path, sep=';')
    
    df = load_data('Tesk_Task___Mail_Retention.csv')
    
    with st.expander("Ukážka datasetu"):
        st.dataframe(df.head())
    st.divider()
    
    # ─────── Фільтри ───────
    st.sidebar.header("Filter")
    
    segment = st.sidebar.selectbox( "Сегмент користувачів",["All", "Buyers", "Non-buyers"],index=0)
    
    st.sidebar.markdown("### Період")
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_from, date_to = st.sidebar.date_input("Діапазон дат",value=(min_date, max_date),min_value=min_date,max_value=max_date)

    
    selected_test = st.sidebar.selectbox("Оберіть тест для аналізу",options=["group_1", "group_2", "group_3", "group_4"],index=0,key="ab_select")
    
    
    
    # ───────Ствоерння фільтрованого датасету ───────
    df_filtered = df.copy()
    
    
    if segment == "Buyers":
        df_filtered = df_filtered[df_filtered['buyer'] == 'Buyer']

    elif segment == "Non-buyers":
        df_filtered = df_filtered[df_filtered['buyer'] == 'Not Buyer']

    
    df_filtered = df_filtered[
        (df_filtered['date'] >= pd.to_datetime(date_from)) &
        (df_filtered['date'] <= pd.to_datetime(date_to))
    ]

    df_ab = df_filtered.copy()                  
    df_ab['variant'] = df_ab[selected_test].fillna("Unknown").astype(str)
    
    test_df    = df_ab[df_ab['variant'] == "Test"].copy()
    control_df = df_ab[df_ab['variant'] == "Control"].copy()
    
    # ========================== A/B TESTING  ==========================
    st.header("A/B-аналіз")
    
    
    # --- ----KPI--------
    st.subheader("Вибірка")
    
    
    total_events = len(df_ab)
    total_users = df_ab['user_id'].nunique() if 'user_id' in df_ab.columns else 0
    

    test_events = len(test_df)
    ctrl_events = len(control_df)
    
    test_users = test_df['user_id'].nunique() if 'user_id' in test_df.columns else 0
    ctrl_users = control_df['user_id'].nunique() if 'user_id' in control_df.columns else 0
    
    buyers_users = (
        df_ab[df_ab['buyer'] == 'Buyer']['user_id'].nunique()
        if 'user_id' in df_ab.columns else 0
    )
    nonbuyers_users = total_users - buyers_users
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users (total)", f"{total_users:,}")
    c2.metric("Events (total)", f"{total_events:,}")
    c3.metric("Users Test", f"{test_users:,}")
    c4.metric("Users Control", f"{ctrl_users:,}")
    c5.metric("Buyers / Non-buyers", f"{buyers_users:,} / {nonbuyers_users:,}")
    
    
    
    #--------Порівняння тест контрол + п-валю-------
    def get_ab_metrics(df):
        if len(df) == 0: return {'Sent': 0, 'Open rate': 0.0, 'CTR': 0.0, 'Conv rate': 0.0}
        sent = len(df)
        opened = df['read_ts'].notna().sum()
        clicked = df['click_ts'].notna().sum()
        converted = (df['total_credits'].fillna(0) > 0).sum()
        return {
            'Sent': sent,
            'Open rate': opened / sent if sent else 0,
            'CTR': clicked / sent if sent else 0,
            'Conv rate': converted / sent if sent else 0
        }
    
    m_test = get_ab_metrics(test_df)
    m_ctrl = get_ab_metrics(control_df)
    
    def chi2_pvalue(test_col, ctrl_col):
        table = np.array([[test_col.sum(), len(test_col)-test_col.sum()],
                          [ctrl_col.sum(), len(ctrl_col)-ctrl_col.sum()]])
        if table.min() == 0: return np.nan
        _, p, _, _ = chi2_contingency(table)
        return p
    
    p_open = chi2_pvalue(test_df['read_ts'].notna(), control_df['read_ts'].notna())
    p_ctr  = chi2_pvalue(test_df['click_ts'].notna(), control_df['click_ts'].notna())
    p_conv = chi2_pvalue(test_df['total_credits'].fillna(0) > 0, control_df['total_credits'].fillna(0) > 0)
    
    ab_table = pd.DataFrame({
        "Метрика": ["Open rate", "CTR", "Conv rate"],
        "Control": [m_ctrl["Open rate"], m_ctrl["CTR"], m_ctrl["Conv rate"]],
        "Test":    [m_test["Open rate"], m_test["CTR"], m_test["Conv rate"]],
        "p-value": [p_open, p_ctr, p_conv]
    })
    
    ab_table["Lift"] = (ab_table["Test"] - ab_table["Control"]) / ab_table["Control"].replace(0, np.nan)
    
    def significance(p, lift):
        if pd.isna(p) or pd.isna(lift): return "—"
        if p < 0.05:
            return "Значуще(позитив)" if lift > 0 else "Значуще (негатив)"
        return "Не значуще"
    
    ab_table["Значущість"] = ab_table.apply(lambda row: significance(row["p-value"], row["Lift"]), axis=1)
    
    st.subheader(f"Порівняння Test vs Control для {selected_test}")
    st.dataframe(ab_table.style.format({
        "Control": "{:.2%}", "Test": "{:.2%}", "Lift": "{:+.1%}", "p-value": "{:.4f}"
    }).background_gradient(subset=["Lift"], cmap="RdYlGn"), use_container_width=True)
    

    
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, ab_table["Control"], width, label="Control", color="#1f77b4")
    ax.bar(x + width/2, ab_table["Test"], width, label="Test", color="#ff7f0e")
    ax.set_ylabel("Rate")
    ax.set_title(f"{selected_test} — Test vs Control")
    ax.set_xticks(x)
    ax.set_xticklabels(ab_table["Метрика"])
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    
    
    #----------ШІ овервю--------
    st.subheader("AI рекомендація по тесту")
    if st.button("Згенерувати висновок та рекомендацію", type="primary"):
        prompt = f"""Ти — старший аналітик email retention. Період фільтра: {date_from} – {date_to}.
    
    Порівняння дня: {today.date()} vs {prev_label}
    
    Open rate: {open_recent:.1%} (Δ {delta_open:+.1%})
    CTR: {ctr_recent:.1%} (Δ {delta_ctr:+.1%})
    Conversion rate: {conv_recent:.1%} (Δ {delta_conv:+.1%})
    Sent (сьогодні): {len(last_day):,}
    Buyers (у фільтрі): {buyers_cnt:,}
    
    Останні 14 днів (динаміка):
    {daily.tail(14)[['sent','open_rate','ctr','conversion_rate']].round(3).to_string()}
    
    Напиши коротке самарі українською (4–6 речень):
    • Що покращилось / погіршилось відносно вчора (або середнього)
    • Де найбільш помітні зміни
    • 2 прості рекомендації наступних кроків
    """  
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        with st.spinner("Groq думає..."):
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7
            )
        st.success(chat.choices[0].message.content)
    
