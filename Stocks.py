import streamlit as st
import pandas as pd
import ta
import matplotlib.pyplot as plt
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import yfinance as yf

def annualize_rets(r,periods_per_years):
    compounded_growth=(1+r).prod()
    n_periods=r.shape[0]
    return (compounded_growth**(periods_per_years/n_periods))-1

import numpy as np
def annualize_vol(r,periods_per_years):
    return r.std()*((periods_per_years)**0.5)

def sharpe_ratio(r,riskfree_rate,periods_per_years):
    rf_per_period=(1+riskfree_rate)**(1/periods_per_years) - 1
    excess_ret=r-rf_per_period
    ann_ex_ret=annualize_rets(excess_ret,periods_per_years)
    ann_vol=annualize_vol(r,periods_per_years)
    return ann_ex_ret/ann_vol

def drawdown(return_series:pd.Series):
    """
    Takes a times series of asset returns
    Computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    percent drawdown
    """
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdowns=(wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth":wealth_index,
        "Peaks":previous_peaks,
        "Drawdown":drawdowns})

def skewness(r):
    demeaned_r= r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**3).mean()
    return exp/(sigma_r**3)

def kurtosis(r):
    demeaned_r= r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**4).mean()
    return exp/(sigma_r**4)

import scipy.stats
def is_normal(r,level=0.01):
    statistic, p_value=scipy.stats.jarque_bera(r)
    return p_value>level
    
    
import numpy as np    
def var_historic(r, level=5,historic=504):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r[-historic:], pd.DataFrame):
        return r[-historic:].aggregate(var_historic, level=level)
    elif isinstance(r[-historic:], pd.Series):
        return -np.percentile(r[-historic:], level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
from scipy.stats import norm
def var_gaussian(r,level=5):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    z=norm.ppf(level/100)
    return -(r.mean()+ z*r.std(ddof=0))



def var_gaussian(r,level=5,modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If modified is True, then the modified VaR is returned, using the Cornish-Fisher modification
    """
    z=norm.ppf(level/100)
    if modified:
        s=skewness(r)
        k=kurtosis(r)
        z= (z+
               (z**2-1)*s/6+
                (z**3-3*z)*(k-3)/24-
                (2*z**3-5*z)*(s**2)/36
           
           
           )
    return -(r.mean()+ z*r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    VaR Historic
    """
    if isinstance(r,pd.Series):
        is_beyond=r<= -var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance (r,pd.DataFrame):
        return r.aggregate(cvar_historic,level=level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
    
def cvar_gaussian(r,level=5):
    is_beyond2=r<=-var_gaussian(r,level=level)
    return -r[is_beyond2].mean()


def summary_stats(r,riskfree_rate=0.03,periods_per_years=52):
    ann_r=r.aggregate(annualize_rets,periods_per_years=periods_per_years)
    ann_vol=r.aggregate(annualize_vol,periods_per_years=periods_per_years)
    ann_sr=r.aggregate(sharpe_ratio,riskfree_rate=riskfree_rate,periods_per_years=periods_per_years)
    dd=r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew=r.aggregate(skewness)
    kurt=r.aggregate(kurtosis)
    cf_var5=r.aggregate(var_gaussian,modified=True)
    hist_cvar5=r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized return": ann_r,
        "Annualized Volatility": ann_vol,
        "Skewness":skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR(5%)":cf_var5,
        "Historic CVaR(5%)":hist_cvar5,
        "Sharpe Ratio":ann_sr,
        "Max Drawdown":dd
    })

ticker=['TTE.PA','EDF.PA','VLA.PA','MCPHY.PA','^FCHI','ORA.PA','ALBPS.PA','NRJ.PA','AIR.PA']
titre=['TOTAL','EDF','VALNEVA','MCPHYENERGY','CAC','ORANGE','BIOPHYTIS','ETF NEW ENERGY','AIRBUS']

action = st.radio(
                "What's kind of analysis do you want to do?",
                ('Single assset', 'Multi assets'))

if action == 'Single assset':
    
    option=st.sidebar.selectbox('Choisir un titre',titre)
    titre_choisi=option
    for i in range(0,len(titre)):
        if titre_choisi==titre[i]:
            ticker_choisi=ticker[i]
    st.title('Analyse'+' '+titre_choisi)
    
    import datetime
    today=datetime.datetime.today()
    before=today-datetime.timedelta(days=(5*365))
    start_date=st.sidebar.date_input('Date de début',before)
    end_date=st.sidebar.date_input('Date de fin',today)
    if start_date<end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must fall after start date.')
        
    df=yf.download(ticker_choisi,start=start_date,end=end_date,progress=False)
    returns=df[['Close']].pct_change().dropna()
    perf=summary_stats(returns,riskfree_rate=0.01,periods_per_years=252)

    df['SMA20']= ta.trend.sma_indicator(close=df['Close'], window=20)
    df['SMA50']= ta.trend.sma_indicator(close=df['Close'], window=50)
    df['SMA100']= ta.trend.sma_indicator(close=df['Close'], window=100)
    df['SMA150']= ta.trend.sma_indicator(close=df['Close'], window=150)
    SMA=df[['Close','SMA20','SMA50','SMA100','SMA150']]

    ytd=datetime.datetime.today().year-1
    last_day=datetime.date(ytd,12,31)
    df2=yf.download(ticker_choisi,start=last_day,end=datetime.datetime.today(),progress=False)
    returns2=df2[['Close']].pct_change().dropna()
    perf_ytd=(df2['Close'][-1]-df2['Close'][str(ytd)][-1])/df2['Close'][str(ytd)][-1]

    mtd=datetime.datetime.today().month-1
    perf_mtd=(df2['Close'][-1]-df2['Close'][str(ytd+1)+'-'+str(mtd)][-1])/df2['Close'][str(ytd+1)+'-'+str(mtd)][-1]

    # Moving Average Convergence Divergence
    macd = MACD(df['Close']).macd()

    # Resistence Strength Indicator
    rsi = RSIIndicator(df['Close']).rsi()

    ###################
    # Set up main app #
    ###################
    col1, col2, col3 = st.columns(3)
    col1.metric('Dernier prix',round(df2['Close'][-1],2), "{:.2%}".format(float(returns2.values[-1])))
    col2.metric('Perf YTD',"{:.2%}".format(perf_ytd))
    col3.metric('Perf MTD',"{:.2%}".format(perf_mtd))

    # Plot the prices and the simple moving average
    import plotly.express as px
    st.write('Stock SMA')
    fig = px.line(SMA)
    fig.update_layout(
        autosize=False,
        width=800,
        height=450,
        margin=dict(
            l=0,
            r=100,
            b=50,
            t=50,
            pad=0)
    )
    st.plotly_chart(fig)
    #Candlestick
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=dict(
            l=0,
            r=100,
            b=50,
            t=50,
            pad=0)
    )
    
    st.plotly_chart(fig)

    #Stats importantes sur le titre
    st.write('Risk metrics')
    st.dataframe(perf)

    # Plot MACD
    st.write('Stock Moving Average Convergence Divergence (MACD)')
    st.area_chart(macd)

    # Plot RSI
    st.write('Stock RSI ')
    st.line_chart(rsi)
    if rsi[-1]>=70:
        st.markdown('Attention le titre est suracheté')
    elif rsi[-1]<=30:
        st.markdown(titre_chosi+'est sous acheté')
    
    # Data of recent days
    st.write('Recent data ')
    st.dataframe(df2.tail(10))
    
elif action=='Multi assets':
    titres=[]
    for i in range(0,len(titre)):
        if titre[i]!='CAC':
            titres.append(titre[i])
    ptf=st.multiselect('Choose your stocks to build your portfolio',titre,titres)
    tickers=[]
    for i in range(0,len(ptf)):
        if ptf[i] in titre:
            tickers.append(ticker[titre.index(ptf[i])])
    
    #On télécharge les data pour tous les titres du portefeuille
    import datetime
    today=datetime.datetime.today()
    before=today-datetime.timedelta(days=(5*365))
    start_date=st.sidebar.date_input('Date de début',before)
    end_date=st.sidebar.date_input('Date de fin',today)
    if start_date<end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must fall after start date.')
        
    df=yf.download(tickers,start=start_date,end=end_date,progress=False)
    returns=df[['Close']].pct_change().dropna()
    
    #Composition du portefeuille
    if st.checkbox('Change the number of stocks'):
        nb_total=st.number_input('Nombre de titres '+ptf[0],value=3)
        nb_edf=st.number_input('Nombre de titres '+ptf[1],value=10)
        nb_valneva=st.number_input('Nombre de titres '+ptf[2],value=35)
        nb_mcphy=st.number_input('Nombre de titres '+ptf[3],value=25)
        nb_orange=st.number_input('Nombre de titres '+ptf[4],value=31)
        nb_bio=st.number_input('Nombre de titres '+ptf[5],value=60)
        nb_etf=st.number_input('Nombre de titres '+ptf[6],value=6)
        nb_airbus=st.number_input('Nombre de titres '+ptf[7],value=5)
        nb=[nb_total,nb_edf,nb_valneva,nb_mcphy,nb_orange,nb_bio,nb_etf,nb_airbus]
    else:
        nb_total=3
        nb_edf=10
        nb_valneva=35
        nb_mcphy=25
        nb_orange=31
        nb_bio=60
        nb_etf=6
        nb_airbus=5
        nb=[nb_total,nb_edf,nb_valneva,nb_mcphy,nb_orange,nb_bio,nb_etf,nb_airbus]
        
    #Calcul de la valeur du portefeuille
    pf_price=df['Close'][tickers]*nb
    pf_price['Valeur portefeuille']=pf_price[pf_price.columns].sum(axis=1)
    
    ytd=datetime.datetime.today().year-1
    last_day=datetime.date(ytd,12,31)
    df2=yf.download(tickers,start=last_day,end=datetime.datetime.today(),progress=False)
    pf_price2=df2['Close'][tickers]*nb
    
    pf_price2['Valeur portefeuille']=pf_price2[pf_price2.columns].sum(axis=1)
    pf_price2=pf_price2.dropna()
    import math
    na=[]
    for j in range(0,len(pf_price2.columns)):
        na.append(int(math.isnan(pf_price2.iloc[-1,j])))
        if sum(na)>0:
            vl_ptf=pf_price2.iloc[-2][8]
        else:
            vl_ptf=pf_price2.iloc[-1][8]
            
    returns2=df2[['Close']].pct_change().dropna()
    perf_ytd=(vl_ptf-pf_price2['Valeur portefeuille'][str(ytd)][-1])/pf_price2['Valeur portefeuille'][str(ytd)][-1]
    mtd=datetime.datetime.today().month-1
    perf_mtd=(vl_ptf-pf_price2['Valeur portefeuille'][str(ytd+1)+'-'+str(mtd)][-1])/pf_price2['Valeur portefeuille'][str(ytd+1)+'-'+str(mtd)][-1]

    col1, col2, col3 = st.columns(3)
    col1.metric('Valeur du portefeuille',round(vl_ptf,2), round(vl_ptf-pf_price2['Valeur portefeuille'][-2],2))
    col2.metric('Perf YTD',"{:.2%}".format(perf_ytd))
    col3.metric('Perf MTD',"{:.2%}".format(perf_mtd))
    
    import plotly.express as px
    fig = px.line(pf_price2['Valeur portefeuille'])
    fig.update_layout(
        autosize=False,
        width=800,
        height=450,
        margin=dict(
            l=0,
            r=100,
            b=50,
            t=50,
            pad=0)
    )
    st.plotly_chart(fig)
    st.write('Last % change')
    st.dataframe(returns2.iloc[-1]*100)

    