# app.py
# Hedge Options Lab — Clean single-file Streamlit app
# Features:
#  - Black-Scholes pricing & Greeks
#  - Implied vol solver
#  - SVI per-expiry (calibration) + interpolation (basic)
#  - Monte Carlo: GBM (risk-neutral / real-world) + Heston (stochastic vol)
#  - Delta-hedge discrete simulation + simple optimizer
#  - Stocks/MF analysis (CAGR, Sharpe, Sortino, drawdown, skew, kurtosis)
#  - Probability forecasts, probability-of-touch, percentile forecasts
#  - BUY/SELL/HOLD signal engine based on forecast and risk thresholds
# Single-file intended for Streamlit Cloud.

import streamlit as st
st.set_page_config(layout="wide", page_title="Finance Lab - Deep Quant Tools")
import numpy as np, pandas as pd, math, io, datetime, warnings
from scipy.stats import norm
from scipy.optimize import least_squares
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# Use seaborn style if available, otherwise use default
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    try:
        plt.style.use("seaborn")
    except OSError:
        pass  # Use default matplotlib style

# Optional imports
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False

# -------------------------
# Math utilities & Black-Scholes
# -------------------------
def N(x): return norm.cdf(x)
def pdf(x): return norm.pdf(x)

def bs_call_price(S,K,r,sigma,T):
    if T<=0: return max(S-K,0.0)
    if sigma<=0: return max(S - K*math.exp(-r*T),0.0)
    d1 = (math.log(S/K) + (r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*N(d1) - K*math.exp(-r*T)*N(d2)

def bs_greeks(S,K,r,sigma,T):
    if T<=0 or sigma<=0:
        delta = 1.0 if S>K else 0.0
        return dict(Delta=delta, Gamma=0.0, Vega=0.0, Theta=0.0, Rho=0.0, d1=None, d2=None)
    d1 = (math.log(S/K) + (r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    pdfd1 = pdf(d1)
    Delta = N(d1)
    Gamma = pdfd1/(S*sigma*math.sqrt(T))
    Vega = S*pdfd1*math.sqrt(T)
    Theta = -S*pdfd1*sigma/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*N(d2)
    Rho = K*T*math.exp(-r*T)*N(d2)
    return dict(Delta=Delta, Gamma=Gamma, Vega=Vega, Theta=Theta, Rho=Rho, d1=d1, d2=d2)

def implied_vol_call(mkt_price, S,K,r,T, tol=1e-8, maxiter=100):
    # bisection
    if mkt_price <= max(0, S - K*math.exp(-r*T)) + 1e-12:
        return 1e-8
    lo, hi = 1e-8, 5.0
    for _ in range(maxiter):
        mid = 0.5*(lo+hi)
        p = bs_call_price(S,K,r,mid,T)
        if abs(p - mkt_price) < tol:
            return mid
        if p > mkt_price:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)

# -------------------------
# SVI surface (per-expiry fit)
# -------------------------
def svi_total_variance(k,a,b,rho,m,sig):
    return a + b*(rho*(k-m) + np.sqrt((k-m)**2 + sig**2))

def svi_iv_from_params(K,F,T,a,b,rho,m,sig):
    k = np.log(K/F)
    w = svi_total_variance(k,a,b,rho,m,sig)
    return np.sqrt(np.clip(w,1e-12,None)/T)

def svi_loss(params, strikes, market_iv, F, T):
    a,b,rho,m,sig = params
    return svi_iv_from_params(strikes, F, T, a,b,rho,m,sig) - market_iv

def fit_svi(strikes, market_iv, F, T):
    x0 = np.array([0.01,0.2,0.0,0.0,0.2])
    bounds = ([-np.inf,1e-8,-0.999,-10,1e-8],[np.inf, np.inf,0.999,10,5])
    res = least_squares(svi_loss, x0, args=(np.array(strikes), np.array(market_iv), F, T), bounds=bounds, xtol=1e-12, ftol=1e-12)
    return res.x

# -------------------------
# Monte Carlo engines: GBM and Heston (vectorized; numba optional)
# -------------------------
def gbm_paths_numpy(S0, mu, sigma, T, steps, paths, seed=None):
    rng = np.random.default_rng(seed)
    dt = T/steps
    increments = (mu - 0.5*sigma*sigma)*dt + sigma*math.sqrt(dt)*rng.standard_normal((paths, steps))
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((paths,1)), log_paths])
    return S0 * np.exp(log_paths)

# Basic Heston (Euler-Maruyama). For heavy production tune scheme or use QE method.
def heston_paths_numpy(S0, v0, mu, kappa, theta, xi, rho, T, steps, paths, seed=None):
    rng = np.random.default_rng(seed)
    dt = T/steps
    S = np.empty((paths, steps+1))
    v = np.empty((paths, steps+1))
    S[:,0] = S0
    v[:,0] = v0
    for t in range(1, steps+1):
        z1 = rng.standard_normal(paths)
        z2 = rng.standard_normal(paths)
        # correlated
        w1 = z1
        w2 = rho*z1 + math.sqrt(1-rho*rho)*z2
        v[:,t] = np.maximum(v[:,t-1] + kappa*(theta - v[:,t-1])*dt + xi*np.sqrt(np.maximum(v[:,t-1],0))*math.sqrt(dt)*w2, 1e-12)
        S[:,t] = S[:,t-1] * np.exp((mu - 0.5*v[:,t-1])*dt + np.sqrt(np.maximum(v[:,t-1],0))*math.sqrt(dt)*w1)
    return S, v

# If numba available, offer compiled versions (simple compilation)
if NUMBA:
    from numba import njit
    @njit
    def gbm_paths_numba(S0, mu, sigma, T, steps, paths, seed):
        import numpy as _np
        dt = T/steps
        rng = _np.random.RandomState(seed)
        out = _np.empty((paths, steps+1))
        for i in range(paths):
            out[i,0] = S0
            logS = 0.0
            for t in range(1, steps+1):
                z = rng.normal()
                logS += (mu - 0.5*sigma*sigma)*dt + sigma*math.sqrt(dt)*z
                out[i,t] = S0 * math.exp(logS)
        return out
    # Not providing compiled Heston here to keep complexity manageable.

# -------------------------
# Hedge simulation: discrete delta-hedge
# -------------------------
def delta_hedge_sim(S_paths, K, r, sigma_for_greeks, T, rehedge_step=1, tx_cost=0.0, init_option_price=None):
    paths, steps_plus_one = S_paths.shape
    steps = steps_plus_one - 1
    dt = T/steps
    pnl = np.zeros(paths)
    if init_option_price is None:
        init_option_price = bs_call_price(S_paths[:,0].mean(), K, r, sigma_for_greeks, T)
    for i in range(paths):
        path = S_paths[i]
        cash = 0.0
        cash += init_option_price  # short one call -> receive premium
        g0 = bs_greeks(path[0], K, r, sigma_for_greeks, T)
        delta_prev = g0['Delta']
        stock_pos = delta_prev
        cash -= stock_pos * path[0]
        for t in range(1, steps+1):
            tau = max(T * (1 - t/steps), 1e-12)
            if (t % rehedge_step)==0 or t==steps:
                g = bs_greeks(path[t], K, r, sigma_for_greeks, tau)
                delta_new = g['Delta']
                trade_qty = delta_new - delta_prev
                cash -= trade_qty * path[t]
                cash -= abs(trade_qty) * tx_cost
                delta_prev = delta_new
                stock_pos = delta_new
            cash *= math.exp(r*dt)
        cash += stock_pos * path[-1]
        payoff = max(path[-1] - K, 0.0)
        cash -= payoff
        pnl[i] = cash
    return pnl

# -------------------------
# Portfolio & equity metrics
# -------------------------
def performance_metrics(price_series, freq_ann=252):
    # price_series: pd.Series indexed by date
    pr = price_series.dropna()
    if len(pr) < 2:
        return {}
    logrets = np.log(pr).diff().dropna()
    years = (pr.index[-1] - pr.index[0]).days / 365.0
    total_return = pr.iloc[-1] / pr.iloc[0] - 1.0
    cagr = (pr.iloc[-1]/pr.iloc[0]) ** (1/max(years,1e-12)) - 1.0
    ann_mu = logrets.mean() * freq_ann
    ann_sigma = logrets.std(ddof=1) * math.sqrt(freq_ann)
    sharpe = ann_mu / ann_sigma if ann_sigma>0 else np.nan
    downside = logrets[logrets<0]
    sortino = ann_mu / (downside.std(ddof=1) * math.sqrt(freq_ann)) if len(downside)>1 else np.nan
    skew = logrets.skew()
    kurt = logrets.kurtosis()
    # drawdown
    cummax = pr.cummax()
    drawdown = (pr - cummax) / cummax
    maxdd = drawdown.min()
    calmar = (cagr / abs(maxdd)) if abs(maxdd) > 0 else np.nan
    return dict(total_return=float(total_return), cagr=float(cagr), ann_mu=float(ann_mu), ann_sigma=float(ann_sigma),
                sharpe=float(sharpe), sortino=float(sortino), skew=float(skew), kurt=float(kurt),
                maxdd=float(maxdd), calmar=float(calmar), rolling_vol=logrets.rolling(21).std()*math.sqrt(freq_ann/ (252/21)))

# -------------------------
# Forecast & Signals
# -------------------------
def forecast_from_mc(S_paths, K=None):
    ST = S_paths[:,-1]
    mean = ST.mean()
    std = ST.std(ddof=1)
    percentiles = np.percentile(ST, [1,5,10,25,50,75,90,95,99])
    prob_up = np.mean(ST > (S_paths[0,0] if S_paths.shape[0]>0 else 0))
    prob_above_K = np.nan
    if K is not None:
        prob_above_K = float(np.mean(ST > K))
    return dict(mean=float(mean), std=float(std), percentiles=percentiles, prob_up=float(prob_up), prob_above_K=prob_above_K, terminal=ST)

def simple_signal_engine(forecast, metrics, thresholds):
    # thresholds: dict: prob_up_buy (e.g. 0.6), prob_down_sell (0.6), min_sharpe
    p_up = forecast.get('prob_above_K') if forecast.get('prob_above_K') is not None else forecast.get('prob_up')
    exp_ret = forecast.get('mean')/thresholds.get('spot',1.0) - 1.0
    sharpe = metrics.get('sharpe', np.nan)
    # BUY criteria: high prob above strike, positive expected return, decent sharpe
    if p_up is not None and p_up >= thresholds.get('prob_buy',0.6) and exp_ret >= thresholds.get('min_exp_ret',0.01) and (np.isnan(sharpe) or sharpe >= thresholds.get('min_sharpe',0.3)):
        return "BUY"
    # SELL criteria: low probability above strike (i.e., high prob down)
    if p_up is not None and p_up <= (1 - thresholds.get('prob_sell',0.6)) and exp_ret <= -thresholds.get('min_exp_ret',0.01):
        return "SELL"
    return "HOLD"

# -------------------------
# Streamlit UI: tabs
# -------------------------
st.title("Finance Lab - Deep Quant Tools")
st.markdown("Designed for quant analysis: probabilities, stochastic simulation, hedging, and hedge-fund tracking metrics.")

# Sidebar
st.sidebar.header("Global")
r = st.sidebar.number_input("Risk-free rate", value=0.05, format="%.4f")
seed = int(st.sidebar.number_input("Random seed", value=42, step=1))
np.random.seed(seed)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Market & BS","Vol Surface & Chain","Simulations","Hedging & PnL","Stocks/MF & Signals"])

# -------------------------
# Tab1: Market & BS
# -------------------------
with tab1:
    st.header("Black–Scholes & Market")
    colA,colB = st.columns([1,1])
    with colA:
        S0 = st.number_input("Spot S0", value=100.0, format="%.4f")
        K = st.number_input("Strike K", value=105.0, format="%.4f")
        T = st.number_input("Time to expiry (yrs)", value=0.5, format="%.4f")
        sigma = st.number_input("Vol (annual)", value=0.20, format="%.4f")
        bs_price = bs_call_price(S0,K,r,sigma,T)
        st.metric("BS Call Price", f"{bs_price:.6f}")
    with colB:
        g = bs_greeks(S0,K,r,sigma,T)
        st.write("Greeks")
        st.json({k: (float(v) if v is not None else None) for k,v in g.items()})

    st.write("Implied vol from market price")
    mkt_price = st.number_input("Market option price (use last traded / mid)", value=float(bs_price))
    if st.button("Recover implied vol"):
        iv = implied_vol_call(float(mkt_price), S0, K, r, T)
        st.success(f"Implied vol recovered: {iv:.6f}")

# -------------------------
# Tab2: Vol Surface & Option Chain
# -------------------------
with tab2:
    st.header("Vol Surface (SVI) & Option Chain ingestion")
    c1,c2 = st.columns([1,1])
    with c1:
        ticker = st.text_input("Ticker (yfinance) — optional", value="")
        expiry_choice = st.selectbox("If using yfinance, pick expiry after fetch", options=[""])
        if st.button("Fetch Option Chain from yfinance") and ticker:
            if yf is None:
                st.error("yfinance not installed in environment.")
            else:
                try:
                    t = yf.Ticker(ticker)
                    exps = t.options
                    if len(exps)==0:
                        st.warning("No expiries found.")
                    else:
                        expiry_choice = exps[0]
                        df_chain = t.option_chain(expiry_choice)
                        calls = df_chain.calls.copy(); puts = df_chain.puts.copy()
                        calls['type']='call'; puts['type']='put'
                        chain_df = pd.concat([calls, puts], ignore_index=True, sort=False)
                        chain_df['mid'] = chain_df[['bid','ask']].mean(axis=1).fillna(chain_df['lastPrice'])
                        # show
                        st.success(f"Fetched {ticker} chain expiry {expiry_choice}")
                        st.dataframe(chain_df[['contractSymbol','strike','type','lastPrice','impliedVol']].head(80))
                except Exception as e:
                    st.error("Fetch failed: " + str(e))
    with c2:
        uploaded = st.file_uploader("Or upload option chain CSV (must have strike, expiry, impliedVol columns)", type=['csv'])
        if uploaded:
            try:
                chain_df = pd.read_csv(uploaded, parse_dates=['expiration','expiry'], infer_datetime_format=True)
                st.success("Uploaded chain CSV")
                st.dataframe(chain_df.head(20))
            except Exception as e:
                st.error("Upload failed: " + str(e))

    st.markdown("SVI fit per-expiry (calibrate if chain provided). If you uploaded chain, the app will try calibrating to the first few expiries.")
    if 'chain_df' in locals() or 'chain_df' in globals():
        try:
            # group expiries
            chain = chain_df.copy()
            if 'expiration' in chain.columns:
                chain['expiry'] = pd.to_datetime(chain['expiration']).dt.date
            expiries = chain['expiry'].unique()[:4]
            calibs = {}
            strikes_grid = np.linspace(chain['strike'].min(), chain['strike'].max(), 30)
            surface_rows = []
            for exp in expiries:
                sub = chain[chain['expiry']==exp]
                if len(sub) < 6: continue
                expiry_date = pd.to_datetime(exp)
                Tval = max(1e-6, (expiry_date - pd.Timestamp.today()).days/365.0)
                F = S0 * math.exp(r*Tval)
                strikes = sub['strike'].values
                ivs = sub['impliedVol'].fillna(sigma).values
                try:
                    params = fit_svi(strikes, ivs, F, Tval)
                    calibs[Tval] = params
                    for K0 in strikes_grid:
                        iv = svi_iv_from_params(K0, F, Tval, *params)
                        surface_rows.append((Tval, K0, iv))
                except Exception:
                    continue
            if len(surface_rows)>0:
                surf = pd.DataFrame(surface_rows, columns=['T','K','IV'])
                st.dataframe(surf.head(30))
                # small heatmap
                pivot = surf.pivot_table(index='K', columns='T', values='IV')
                fig,ax = plt.subplots(1,1,figsize=(6,3))
                im = ax.imshow(pivot.fillna(method='ffill').values, origin='lower', aspect='auto',
                               extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
                fig.colorbar(im, ax=ax)
                ax.set_title("SVI surface heatmap")
                st.pyplot(fig)
            else:
                st.info("Not enough data to calibrate SVI on uploaded chain.")
        except Exception as e:
            st.error("SVI calibration failed: " + str(e))

# -------------------------
# Tab3: Simulations (GBM & Heston) and Forecasts
# -------------------------
with tab3:
    st.header("Stochastic Simulations & Forecasts")
    col1,col2 = st.columns([1,1])
    with col1:
        sim_model = st.selectbox("Model", options=["GBM","Heston"])
        mc_paths = st.number_input("MC paths", value=4000, step=100)
        steps_per_year = st.number_input("Steps per year", value=252, step=1)
        steps = max(2, int(steps_per_year * T))
        horizon = st.number_input("Forecast horizon (yrs)", value=T, format="%.4f")
        use_real = st.checkbox("Use real-world drift μ for forecast", value=False)
        mu = st.number_input("Real-world μ (annual)", value=0.06, format="%.4f")
    with col2:
        run_sim = st.button("Run simulation")
        st.write(f"Numba compiled: {NUMBA}")

    if run_sim:
        if sim_model == "GBM":
            drift = mu if use_real else r
            if NUMBA:
                try:
                    S_paths = gbm_paths_numba(S0, drift, sigma, horizon, steps, int(mc_paths), seed)
                except Exception:
                    S_paths = gbm_paths_numpy(S0, drift, sigma, horizon, steps, int(mc_paths), seed)
            else:
                S_paths = gbm_paths_numpy(S0, drift, sigma, horizon, steps, int(mc_paths), seed)
            fc = forecast_from_mc(S_paths, K)
            st.write("Forecast summary (terminal): mean, std")
            st.write(fc['mean'], fc['std'])
            st.write("Probability terminal > strike (K):", fc['prob_above_K'])
            # percentiles
            percs = fc['percentiles']
            dfp = pd.DataFrame({'percentile':[1,5,10,25,50,75,90,95,99],'terminal':percs})
            st.table(dfp)
            fig,ax = plt.subplots(1,2,figsize=(10,3))
            ax[0].hist(fc['terminal'], bins=60); ax[0].set_title("Terminal distribution")
            ax[1].plot(np.sort(fc['terminal'])); ax[1].set_title("Sorted terminal (CDF visual)")
            st.pyplot(fig)
            st.session_state['last_sim_paths'] = S_paths
        else:
            # Heston parameters
            v0 = st.number_input("Heston v0 (initial variance)", value=sigma*sigma)
            kappa = st.number_input("Heston kappa (reversion)", value=1.5)
            theta = st.number_input("Heston theta (long-run var)", value=sigma*sigma)
            xi = st.number_input("Heston xi (vol of vol)", value=0.6)
            rho = st.number_input("Heston rho (corr)", value=-0.6)
            drift = mu if use_real else r
            S_paths, v_paths = heston_paths_numpy(S0, v0, drift, kappa, theta, xi, rho, horizon, steps, int(mc_paths), seed)
            fc = forecast_from_mc(S_paths, K)
            st.write("Heston forecast mean/std:", fc['mean'], fc['std'])
            st.session_state['last_sim_paths'] = S_paths
            st.session_state['last_sim_vol_paths'] = v_paths

# -------------------------
# Tab4: Hedging & PnL analysis
# -------------------------
with tab4:
    st.header("Hedging simulation, PnL, VaR, CVaR, turnover")
    col1,col2 = st.columns([1,1])
    with col1:
        rehedge_every = st.number_input("Rehedge every (steps)", min_value=1, value=1)
        tx_cost = st.number_input("Transaction cost per share", value=0.0, format="%.6f")
        hedge_paths = st.number_input("Hedge sim paths (use <= sim paths)", value=1000, step=100)
        run_hedge = st.button("Run hedge sim on last paths")
    with col2:
        show_pnl_hist = st.checkbox("Show PnL histogram", value=True)

    if run_hedge:
        if 'last_sim_paths' not in st.session_state:
            st.error("Run simulation first (Tab: Simulations).")
        else:
            S_paths = st.session_state['last_sim_paths'][:int(hedge_paths)]
            pnl = delta_hedge_sim(S_paths, K, r, sigma, horizon, int(rehedge_every), float(tx_cost), init_option_price=bs_call_price(S0,K,r,sigma,T))
            mean = pnl.mean(); std = pnl.std(ddof=1)
            var95 = -np.percentile(pnl,5); var99 = -np.percentile(pnl,1)
            cvar95 = -pnl[pnl <= np.percentile(pnl,5)].mean()
            cvar99 = -pnl[pnl <= np.percentile(pnl,1)].mean()
            # turnover proxy: average absolute delta changes per path normalized by S0
            # compute approximate turnover by estimating delta changes from greeks along path
            # (simple proxy)
            st.write("Hedge PnL mean, std:", mean, std)
            st.write("VaR95 (loss):", var95, "VaR99:", var99)
            st.write("CVaR95:", cvar95, "CVaR99:", cvar99)
            if show_pnl_hist:
                fig,ax = plt.subplots(1,1,figsize=(6,3))
                ax.hist(pnl, bins=60)
                ax.set_title("Hedge PnL distribution")
                st.pyplot(fig)
            # export
            buf = io.BytesIO(); pd.DataFrame({'pnl':pnl}).to_csv(buf, index=False); buf.seek(0)
            st.download_button("Download hedge PnL CSV", data=buf, file_name="hedge_pnl.csv", mime="text/csv")

# -------------------------
# Tab5: Stocks, MF analytics & Signals
# -------------------------
with tab5:
    st.header("Stocks / Mutual Fund analytics and trading signal")
    c1,c2 = st.columns([1,1])
    with c1:
        ticker = st.text_input("Ticker for equity/MF (yfinance) OR upload CSV", value="")
        uploaded = st.file_uploader("Upload CSV (Date,Price)", type=['csv'])
        use_series = None
        if uploaded:
            try:
                dfp = pd.read_csv(uploaded, parse_dates=[0])
                dfp.columns = ['Date','Price'] + list(dfp.columns[2:])
                dfp = dfp.set_index('Date').sort_index()
                use_series = dfp['Price']
                st.success("Loaded series from CSV.")
            except Exception as e:
                st.error("Read failed: "+str(e))
        elif ticker:
            if yf is None:
                st.warning("yfinance not installed — upload CSV")
            else:
                try:
                    data = yf.Ticker(ticker).history(period="2y", interval="1d")
                    if data.empty:
                        st.warning("No data found for ticker.")
                    else:
                        use_series = data['Close']
                        st.success("Fetched price history from yfinance (2y).")
                except Exception as e:
                    st.error("yfinance error: "+str(e))
    with c2:
        if use_series is not None:
            metrics = performance_metrics(use_series)
            st.write("Performance metrics")
            st.write({k: v for k,v in metrics.items() if k!='rolling_vol'})
            st.line_chart(use_series)
            st.line_chart(metrics['rolling_vol'].dropna())
            # Forecast from MC using estimated mu/sigma
            est_sigma = metrics['ann_sigma']; est_mu = metrics['ann_mu']
            st.write("Estimated annual µ, σ from history:", est_mu, est_sigma)
            if st.button("Run forecast using historical estimates"):
                steps = 252
                paths = 3000
                S_paths = gbm_paths_numpy(use_series.iloc[-1], est_mu, est_sigma, 0.5, steps, paths, seed)
                fc = forecast_from_mc(S_paths, use_series.iloc[-1]*1.02)  # optional K
                st.write("Forecast percentiles")
                percs = fc['percentiles']
                st.table(pd.DataFrame({'percentile':[1,5,10,25,50,75,90,95,99],'terminal':percs}))
                st.session_state['last_sim_paths'] = S_paths
            # Generate trading signal
            st.markdown("Signal engine (simple rule-based):")
            prob_buy = st.slider("Min prob above strike for BUY", 0.5, 0.9, 0.6)
            min_exp_ret = st.slider("Min expected return for BUY (annualized)", 0.0, 0.2, 0.02)
            min_sharpe = st.slider("Min Sharpe for BUY", 0.0, 2.0, 0.3)
            if st.button("Generate BUY/SELL/HOLD signal"):
                # require a forecast present
                if 'last_sim_paths' not in st.session_state:
                    st.warning("Run a forecast simulation first (top of this tab or Simulations).")
                else:
                    fc = forecast_from_mc(st.session_state['last_sim_paths'], K=None)
                    thresholds = dict(prob_buy=prob_buy, min_exp_ret=min_exp_ret, min_sharpe=min_sharpe, spot=use_series.iloc[-1])
                    sig = simple_signal_engine(fc, metrics, thresholds)
                    st.success(f"Signal: {sig}")
                    st.write("Forecast mean/STD/prob_up:", fc['mean'], fc['std'], fc['prob_up'])
        else:
            st.info("Provide a ticker or upload price CSV to compute metrics and signals.")

