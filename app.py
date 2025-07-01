import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Streamlit page configuration
st.set_page_config(page_title="Trade Summary Dashboard", layout="wide")

# Cache data loading to improve performance
@st.cache_data
def load_data(sdf_path=None, tdf_path=None, sdf_file=None, tdf_file=None):
    try:
        # Load data from file paths or uploaded files
        if sdf_file is not None and tdf_file is not None:
            sdf = pd.read_csv('https://raw.githubusercontent.com/Shailja12326646/MARKET_SENTIMENT_ANALYSIS/main/fear_greed_index.csv')
            tdf = pd.read_csv('https://github.com/Shailja12326646/MARKET_SENTIMENT_ANALYSIS/releases/download/datasetv1/historical_data.csv')

        else:
            sdf = pd.read_csv('https://raw.githubusercontent.com/Shailja12326646/MARKET_SENTIMENT_ANALYSIS/main/fear_greed_index.csv')
            tdf = pd.read_csv('https://github.com/Shailja12326646/MARKET_SENTIMENT_ANALYSIS/releases/download/datasetv1/historical_data.csv')

        
        # Validate required columns
        required_sdf_cols = ['date', 'classification']
        required_tdf_cols = ['Timestamp IST', 'Closed PnL', 'Size USD', 'Side', 'Account', 'Coin', 'Fee']
        missing_sdf_cols = [col for col in required_sdf_cols if col not in sdf.columns]
        missing_tdf_cols = [col for col in required_tdf_cols if col not in tdf.columns]
        
        if missing_sdf_cols or missing_tdf_cols:
            raise ValueError(f"Missing columns in CSV files: fear_greed_index.csv {missing_sdf_cols}, historical_data.csv {missing_tdf_cols}")
        
        # Process dates
        sdf['date'] = pd.to_datetime(sdf['date']).dt.date
        tdf['trade_date'] = pd.to_datetime(tdf['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date
        
        # Merge data
        mdf = pd.merge(tdf, sdf, left_on='trade_date', right_on='date', how='left')
        mdf = mdf[mdf['classification'].notnull()]
        return mdf
    except FileNotFoundError:
        st.error("Error: One or both CSV files not found. Please check the file paths or upload the files.")
        return None
    except ValueError as e:
        st.error(f"Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# File uploader for CSV files
st.sidebar.header("Upload Data Files")
sdf_file = st.sidebar.file_uploader("Upload fear_greed_index.csv", type="csv")
tdf_file = st.sidebar.file_uploader("Upload historical_data.csv", type="csv")

# Try loading from default paths or uploaded files
default_sdf_path = 'https://raw.githubusercontent.com/Shailja12326646/MARKET_SENTIMENT_ANALYSIS/main/fear_greed_index.csv'
default_tdf_path = 'https://github.com/Shailja12326646/MARKET_SENTIMENT_ANALYSIS/releases/download/datasetv1/historical_data.csv'

if sdf_file and tdf_file:
    mdf = load_data(sdf_file=sdf_file, tdf_file=tdf_file)
else:
    mdf = load_data(sdf_path=default_sdf_path, tdf_path=default_tdf_path)

# Check if data loaded successfully
if mdf is None:
    st.stop()

# Title
st.title("Trade Summary Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select a Module", [
    "Average PnL by Sentiment",
    "Trade Size Analysis",
    "Daily Trade Summary",
    "Trader Performance",
    "Performance Metrics vs Sentiment"
])


if page == "Average PnL by Sentiment":
    st.header("Average Closed PnL by Sentiment")
    
    # Bar plot
    x1 = mdf.groupby('classification')['Closed PnL'].mean().reset_index()
    x1 = x1.sort_values(by='Closed PnL', ascending=False)

# Plotly bar chart
    fig = px.bar(
        x1,
        x='classification',
        y='Closed PnL',
        color='classification',
        title='Average Closed PnL by Sentiment',
        labels={'classification': 'Market Sentiment', 'Closed PnL': 'Avg Closed PnL'},
        text='Closed PnL'
)

# Optional formatting
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        xaxis_title='Market Sentiment',
        yaxis_title='Average Closed PnL',
        showlegend=False,
        bargap=0.4,
        height=400,
        width=700
    )

# Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    # Pie chart
    fig = px.pie(
    x1,
    names='classification',
    values='Closed PnL',
    title='Closed PnL by Classification',
    hole=0,  # set to 0 for full pie; try 0.3 for donut
)

# Optional: Customize layout
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(
        height=400,
        width=400,
        showlegend=True
    )

# Display in Streamlit
    st.plotly_chart(fig, use_container_width=False)
    
    # Clustered bar chart
    x2 = mdf.groupby(['classification', 'Side'])['Closed PnL'].mean().unstack().round(2)
    plot_df = x2.reset_index().melt(id_vars='classification', value_vars=['BUY', 'SELL'], var_name='Trade Side', value_name='Average Closed PnL')
    fig = px.bar(plot_df, x='classification', y='Average Closed PnL', color='Trade Side', barmode='group',
                 title='Average PnL by Sentiment and Trade Side', labels={'classification': 'Market Sentiment'})
    fig.update_layout(bargap=0.3)
    st.plotly_chart(fig)

# Module 2: Trade Size Analysis
elif page == "Trade Size Analysis":
    st.header("Average Trade Size by Market Sentiment")
    
    x3 = mdf.groupby('classification')['Size USD'].mean().sort_values(ascending=False).round(2).reset_index()
    fig = px.bar(x3, x='classification', y='Size USD', text='Size USD',
                 title='Average Trade Size by Market Sentiment',
                 labels={'classification': 'Sentiment', 'Size USD': 'Avg Trade Size (USD)'},
                 color='Size USD', color_continuous_scale='Viridis')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(bargap=0.4)
    st.plotly_chart(fig)

# Module 3: Daily Trade Summary
elif page == "Daily Trade Summary":
    st.header("Daily Trade Summary")
    
    unique_dates = sorted(mdf['trade_date'].dropna().unique())
    
    selected_date = st.selectbox("Select a Date", unique_dates, index=len(unique_dates)-1)
    selected_date = pd.to_datetime(selected_date).date()
    df = mdf[mdf['trade_date'] == selected_date]
    
    if df.empty:
        st.write(f"No trades found on {selected_date}.")
    else:
        sentiment = df['classification'].mode()[0]
        total_trades = len(df)
        net_pnl = df['Closed PnL'].sum()
        buy_pnl = df[df['Side'] == 'BUY']['Closed PnL'].sum()
        sell_pnl = df[df['Side'] == 'SELL']['Closed PnL'].sum()
        total_volume = df['Size USD'].sum()
        unique_accounts = df['Account'].nunique()
        most_traded_coin = df['Coin'].mode()[0] if not df['Coin'].isna().all() else "N/A"
        total_fee = df['Fee'].sum()

    
        
        st.markdown(f"""
        <div style='font-size:16px; line-height:1.8'>
        📅 <b>Date:</b> {selected_date} <br>
        🧠 <b>Sentiment:</b> {sentiment} <br>
        👤 <b>Unique Accounts:</b> {unique_accounts} <br>
        🪙 <b>Most Traded Coin:</b> {most_traded_coin} <br>
        📊 <b>Total Trades:</b> {total_trades} <br>
        💼 <b>Total Volume (USD):</b> ${total_volume:,.2f} <br>
        💰 <b>Net PnL:</b> ${net_pnl:,.2f} <br>
        💸 <b>Total Fee Paid:</b> ${total_fee:,.4f} <br><br>

        <b>Side Breakdown:</b><br>
        <br>🔼<b>Buy PnL: ${buy_pnl:,.2f}
        🔽Sell PnL: ${sell_pnl:,.2f}
        </div>
        """, unsafe_allow_html=True)



# Module 4: Trader Performance
elif page == "Trader Performance":
    st.header("Trader Performance Summary")
    
    df = mdf.dropna(subset=['Account', 'classification', 'Closed PnL', 'Size USD'])
    performance_df = df.groupby(['Account', 'classification']).agg(
        total_trades=('Closed PnL', 'count'),
        net_pnl=('Closed PnL', 'sum'),
        avg_pnl=('Closed PnL', 'mean'),
        total_volume=('Size USD', 'sum')
    ).reset_index()
    performance_df['roi_percent'] = (performance_df['net_pnl'] / performance_df['total_volume']) * 100
    performance_df = performance_df.round({'net_pnl': 2, 'avg_pnl': 2, 'total_volume': 2, 'roi_percent': 2})
    
    sentiments = sorted(performance_df['classification'].unique())
    selected_sentiment = st.selectbox("Select Sentiment", sentiments)
    subset = performance_df[performance_df['classification'] == selected_sentiment].sort_values(by='net_pnl', ascending=False).head(10)
    st.dataframe(subset)
    
    # Trader Type Classification
    pivot = performance_df.pivot(index='Account', columns='classification', values='net_pnl').fillna(0)
    greed_sentiments = ['Greed', 'Extreme Greed']
    fear_sentiments = ['Fear', 'Extreme Fear']
    
    def classify_trader(row):
        greed_pnl = sum([row.get(s, 0) for s in greed_sentiments])
        fear_pnl = sum([row.get(s, 0) for s in fear_sentiments])
        all_positive = sum([1 for val in row if val > 0])
        if all_positive >= 3:
            return 'Adaptive'
        elif greed_pnl > 0 and fear_pnl <= 0:
            return 'Trend Follower'
        elif fear_pnl > 0 and greed_pnl <= 0:
            return 'Contrarian'
        else:
            return 'Unclassified'
    
    pivot['trader_type'] = pivot.apply(classify_trader, axis=1)
    classified_traders = pivot.reset_index()[['Account', 'trader_type']]
    types = sorted(classified_traders['trader_type'].unique())
    selected_type = st.selectbox("Select Trader Type", types)
    trader_subset = classified_traders[classified_traders['trader_type'] == selected_type]
    st.dataframe(trader_subset)

# Module 5: Performance Metrics vs Sentiment
elif page == "Performance Metrics vs Sentiment":
    st.header("Performance Metrics vs Sentiment Score")
    
    try:
        df1 = mdf.dropna(subset=['value', 'Closed PnL', 'Size USD'])
        df1['sentiment_score'] = df1['value'].round().astype(int)
        score_groups = df1.groupby('sentiment_score')
        
        results = []
        for score, group in score_groups:
            total_trades = len(group)
            wins = group[group['Closed PnL'] > 0]
            losses = group[group['Closed PnL'] < 0]
            net_pnl = group['Closed PnL'].sum()
            win_rate = len(wins) / total_trades if total_trades else 0
            profit_sum = wins['Closed PnL'].sum()
            loss_sum = -losses['Closed PnL'].sum()
            profit_factor = profit_sum / loss_sum if loss_sum > 0 else np.nan
            mean_pnl = group['Closed PnL'].mean()
            std_pnl = group['Closed PnL'].std()
            sharpe = (mean_pnl / std_pnl) * np.sqrt(total_trades) if std_pnl else np.nan
            cumulative = group['Closed PnL'].cumsum()
            drawdown = (cumulative.cummax() - cumulative).max()
            roi = net_pnl / group['Size USD'].sum() * 100 if group['Size USD'].sum() > 0 else np.nan
            results.append({
                'Sentiment Score': score,
                'Net PnL': round(net_pnl, 2),
                'Win Rate (%)': round(win_rate * 100, 2),
                'Profit Factor': round(profit_factor, 2),
                'Sharpe Ratio': round(sharpe, 2),
                'Max Drawdown': round(drawdown, 2),
                'ROI (%)': round(roi, 2),
                'Trade Count': total_trades
            })
        
        score_df = pd.DataFrame(results).sort_values(by='Sentiment Score')
        st.dataframe(score_df)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=score_df, x='Sentiment Score', y='Net PnL', label='Net PnL', ax=ax)
        sns.lineplot(data=score_df, x='Sentiment Score', y='ROI (%)', label='ROI (%)', ax=ax)
        sns.lineplot(data=score_df, x='Sentiment Score', y='Win Rate (%)', label='Win Rate (%)', ax=ax)
        plt.title("📈 Performance Metrics vs Sentiment Score")
        plt.xlabel("Sentiment Score (0 = Extreme Fear, 100 = Extreme Greed)")
        plt.ylabel("Metric Value")
        plt.grid(True)
        plt.legend()
        st.pyplot(fig)
    except KeyError as e:
        st.error(f"Error: Missing column {str(e)}. Please ensure the 'value' column exists in fear_greed_index.csv.")
    except Exception as e:
        st.error(f"Error in Performance Metrics: {str(e)}")