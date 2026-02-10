import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
from groq import Groq
import os

def run():
    
    
    # -------КОНФІГУРАЦІЯ-----
    st.title("Data App")
    st.markdown("Dashboard для інтерактивного аналізу даних та моніторингу ключових метрик")


    @st.cache_data
    def load_data():
        url = "https://github.com/vi-vi-data/data-app/releases/download/v1/Tesk_Task___Mail_Retention.csv"
        return pd.read_csv(
            url,
            sep=",",
            engine="python",
            low_memory=False
        )

    df = load_data()
    with st.expander("Ukážka datasetu"):
        st.dataframe(df.head())
    st.divider()
    
    #--------БІЧНА ПАНЕЛЬ------
    st.sidebar.header("Filter")
    
    segment = st.sidebar.selectbox(
        "Сегмент користувачів",
        ["All", "Buyers", "Non-buyers"],
        index=1  
    )
    
    
    responses = sorted(df['response'].dropna().unique()) if 'response' in df.columns else []
    selected_responses = st.sidebar.multiselect("Тип івенту",options=responses,default=responses)
    
   
    st.sidebar.markdown("### Активність користувача")
    min_tx = st.sidebar.slider(" Виберіть мін. кількість транзакцій на користувача",min_value=0,max_value=50,value=0,step=1)
    

    st.sidebar.markdown("### Діапазон дат")
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_from, date_to = st.sidebar.date_input("Виберіть бажаний період",value=(min_date, max_date),min_value=min_date,max_value=max_date)
    
    df_filtered = df.copy()
    for col in ['send_ts', 'read_ts', 'click_ts', 'total_credits', 'not_free_credits']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
    
    # ===== Фільтрація і створення нового датасета  =====
    if segment == "Buyers":
        df_filtered = df_filtered[df_filtered['buyer'].astype(str).str.lower().str.contains('buyer')]
    elif segment == "Non-buyers":
        df_filtered = df_filtered[~df_filtered['buyer'].astype(str).str.lower().str.contains('buyer')]
    
    
    if 'response' in df_filtered.columns and selected_responses:
        df_filtered = df_filtered[df_filtered['response'].isin(selected_responses)]
    
    
    if min_tx > 0 and 'user_id' in df_filtered.columns and 'total_credits' in df_filtered.columns:
        tx_per_user = df_filtered.assign(is_tx=df_filtered['total_credits'].fillna(0) > 0).groupby('user_id')['is_tx'].sum()
        
        active_users = tx_per_user[tx_per_user >= min_tx].index
        df_filtered = df_filtered[df_filtered['user_id'].isin(active_users)]
    
    
    df_filtered = df_filtered[(df_filtered['date'] >= pd.to_datetime(date_from)) & (df_filtered['date'] <= pd.to_datetime(date_to)) ]
    
    # ========================== KPI==========================
    kpi_df = df_filtered.copy()
    
    records = len(kpi_df)
    buyers_cnt = kpi_df[kpi_df['buyer'].astype(str).str.lower().str.contains('buyer', na=False) ]['user_id'].nunique()
    
    today = df_filtered['date'].max()
    
    last_day = df_filtered[df_filtered['date'] == today].copy()
    
    prev_day_date = today - pd.Timedelta(days=1)
    prev_day = df_filtered[df_filtered['date'] == prev_day_date].copy()
    
    if len(prev_day) == 0:
        prev_day = df_filtered[df_filtered['date'] < today].copy()
        prev_label = "середнє за всі попередні дні"
    else:
        prev_label = str(prev_day_date.date())
    
    def calc_rate(df_part, col_ts):
        sent = len(df_part)
        opened = df_part[col_ts].notna().sum() if col_ts in df_part.columns else 0
        return opened / sent if sent > 0 else 0
    
    open_recent = calc_rate(last_day, 'read_ts')
    open_prev   = calc_rate(prev_day,  'read_ts')
    ctr_recent  = calc_rate(last_day, 'click_ts')
    ctr_prev    = calc_rate(prev_day,  'click_ts')
    conv_recent = (last_day['total_credits'].fillna(0) > 0).sum() / len(last_day) if len(last_day) else 0
    conv_prev   = (prev_day['total_credits'].fillna(0) > 0).sum() / len(prev_day) if len(prev_day) else 0
    
    delta_open = open_recent - open_prev
    delta_ctr  = ctr_recent  - ctr_prev
    delta_conv = conv_recent - conv_prev
    
    # === Вивід KPI ===
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Записи", f"{records:,}")
    col2.metric("Покупці", f"{buyers_cnt:,}")
    col3.metric("Відсоток відкриттів", f"{open_recent:.1%}", delta=f"{delta_open:+.1%}")
    col4.metric("CTR (клік-рейт)", f"{ctr_recent:.1%}", delta=f"{delta_ctr:+.1%}")
    col5.metric("Конверсія", f"{conv_recent:.1%}", delta=f"{delta_conv:+.1%}")

    
    
    
    #------Перша секція: Загальні дані------
    
    # агрегуємо по днях
    daily = (
        df_filtered
        .groupby('date')
        .agg(
            sent=('delivery_id', 'count'),                
            opened=('read_ts', lambda x: x.notna().sum()),
            clicked=('click_ts', lambda x: x.notna().sum()),
            converted=('total_credits', lambda x: (x.fillna(0) > 0).sum())
        )
        .sort_index()
    )
    
    
    daily['open_rate'] = daily['opened'] / daily['sent']
    daily['ctr'] = daily['clicked'] / daily['sent']
    daily['conversion_rate'] = daily['converted'] / daily['sent']
    
    
    daily = daily.fillna(0)
    
    
    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(daily.index, daily['sent'], color='red')
    ax1.set_ylabel("Надіслано (події)")
    ax1.set_xlabel("Дата")
    ax1.set_title("Email-активність у часі (з урахуванням фільтрів)")
    
    ax2 = ax1.twinx()
    ax2.plot(daily.index, daily['open_rate'])
    ax2.set_ylabel("Відсоток відкриттів")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    
    
    tmp = df_filtered.copy()
    buyer_str = tmp['buyer'].astype(str).str.strip().str.lower()
    tmp['is_buyer'] = buyer_str.eq('buyer')
    tmp['is_tx'] = tmp['total_credits'].fillna(0) > 0
    users_by_segment = tmp.groupby('is_buyer')['user_id'].nunique()
    buyers_users = users_by_segment.get(True, 0)
    nonbuyers_users = users_by_segment.get(False, 0)
    tx_per_user = tmp.groupby('user_id')['is_tx'].sum()

    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].bar(['Непокупці', 'Покупці'], [nonbuyers_users, buyers_users])
    axes[0].set_title('Унікальні користувачі за сегментами (після фільтрів)')
    axes[0].set_ylabel('Кількість користувачів')
    
   
    axes[1].hist(tx_per_user, bins=20)
    axes[1].set_title('Кількість транзакцій на користувача (після фільтрів)')
    axes[1].set_xlabel('Транзакцій на користувача')
    axes[1].set_ylabel('Кількість користувачів')
    
    plt.tight_layout()
    st.pyplot(fig)

    
    # --- Друга секція: Message types overview ---
    
    st.header("Огляд типів повідомлень")
    msg_counts = df_filtered['response'].value_counts()
    
    unread_used = msg_counts.get('unread_used_message', 0)
    unread_unused = msg_counts.get('unread_unused_message', 0)
    welcome = msg_counts.get('welcome_message', 0)
    
    c1, c2, c3 = st.columns(3)
    
    c1.metric("Непрочитане повідомлення (повторне)",f"{unread_used:,}".replace(",", " "))
    c2.metric("Непрочитане повідомлення (перше)",f"{unread_unused:,}".replace(",", " "))
    c3.metric("Вітальне повідомлення",f"{welcome:,}".replace(",", " "))
    
    
    counts = df_filtered['response'].value_counts()
    
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    
    wedges, texts, autotexts = ax.pie(counts.values,autopct='%1.1f%%',startangle=90,textprops={'fontsize': 8})
    ax.set_title("Частка повідомлень", fontsize=10)
    ax.legend(wedges,counts.index,loc="center left",bbox_to_anchor=(1, 0.5),fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    
    
    
    # ---Третя секція: Message performance: First vs Repeat---
    st.header("Ефективність повідомлень")
    
    first_msg = df_filtered[df_filtered['response'] == 'unread_unused_message'].copy()
    repeat_msg = df_filtered[df_filtered['response'] == 'unread_used_message'].copy()
    
    def email_metrics(df_part):
        sent = len(df_part)
        opened = df_part['read_ts'].notna().sum()
        clicked = df_part['click_ts'].notna().sum()
        converted = (df_part['total_credits'].fillna(0) > 0).sum()
    
        return {
            'Sent': sent,
            'Opened': opened,
            'Open rate': opened / sent if sent else 0,
            'Clicked': clicked,
            'CTR': clicked / sent if sent else 0,
            'CTOR': clicked / opened if opened else 0,
            'Conversion rate': converted / sent if sent else 0
        }
    
    first_metrics = email_metrics(first_msg)
    repeat_metrics = email_metrics(repeat_msg)
    
    metrics_table = pd.DataFrame({
        "Метрика": list(first_metrics.keys()),
        "Перше повідомлення": list(first_metrics.values()),
        "Повторне повідомлення": [repeat_metrics[k] for k in first_metrics.keys()]
    })
    
    
    rate_rows = ["Open rate", "CTR", "CTOR", "Conversion rate"]
    
    metrics_table["Перше повідомлення"] = metrics_table.apply(
        lambda r: f"{r['Перше повідомлення']:.1%}" if r["Метрика"] in rate_rows
        else f"{int(r['Перше повідомлення']):,}".replace(",", " "),
        axis=1
    )
    
    metrics_table["Повторне повідомлення"] = metrics_table.apply(
        lambda r: f"{r['Повторне повідомлення']:.1%}" if r["Метрика"] in rate_rows
        else f"{int(r['Повторне повідомлення']):,}".replace(",", " "),
        axis=1
    )
    
    st.dataframe(metrics_table, use_container_width=True)
    

    labels = ["First", "Repeat"]
    metrics = {
        "Open rate": [first_metrics["Open rate"], repeat_metrics["Open rate"]],
        "CTR": [first_metrics["CTR"], repeat_metrics["CTR"]],
    }
    
    x = np.arange(len(labels))
    w = 0.35
    
    fig, ax = plt.subplots(figsize=(3, 3))
    for i, (name, values) in enumerate(metrics.items()):
        ax.bar(x + (i - 0.5) * w, values, w, label=name)
    
    ax.set(xticks=x, xticklabels=labels, ylabel="Rate", title="Open rate vs CTR")
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    
       
    
    # --- СЕКЦІЯ ЧОТИРИ Rule erfomance---
    
    st.header("Ефективність сегментів логіки ")
    
    rule_stats = (df_filtered.groupby('rule').agg(sent=('rule', 'count'),
                                                  opened=('read_ts', lambda x: x.notna().sum()),
                                                  clicked=('click_ts', lambda x: x.notna().sum()),
                                                  converted=('total_credits', lambda x: (x.fillna(0) > 0).sum())
                                                 ).reset_index())
    
    rule_stats['open_rate'] = rule_stats['opened'] / rule_stats['sent']
    rule_stats['ctr'] = rule_stats['clicked'] / rule_stats['sent']
    rule_stats['conversion_rate'] = rule_stats['converted'] / rule_stats['sent']
    
   
    rule_stats_f = rule_stats.copy()
    
   
    show_cols = ['rule', 'sent', 'open_rate', 'ctr', 'conversion_rate']
    rule_table = rule_stats_f[show_cols].sort_values('ctr', ascending=False).copy()
    
    
    for c in ['open_rate', 'ctr', 'conversion_rate']:
        rule_table[c] = rule_table[c].map(lambda v: f"{v:.2%}")
    
    st.subheader("Table")
    st.dataframe(rule_table, use_container_width=True)
    
   
    st.subheader("CTR ")
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(rule_stats_f['rule'], rule_stats_f['ctr'])
    ax.set_ylabel("CTR")
    ax.set_xlabel("Rule")
    ax.set_title("CTR by rule")
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)
    
    
    #------П"ята Секція: ШІ овервю------------
    st.header("🧠 AI Summary по періоду")
    
    if st.button("🔥 Згенерувати AI самарі", type="primary"):
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
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
    
        with st.spinner("Groq думає..."):
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7
            )
        st.success(chat.choices[0].message.content)
    
                
  
    
 
