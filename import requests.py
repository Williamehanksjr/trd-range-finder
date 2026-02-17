import requests
from datetime import datetime, timedelta
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
try:
    import yfinance as yf
except Exception:
    yf = None

# ================= CONFIG =================
INTERVAL = 1
WINDOW_SECONDS = 60 * 60
SYMBOL = "/MCL"  # "/MCL", "MCL=F", "/ES", "GC=F", "BTC-USD"
LINE_COLOR = "blue"
BTC_INVESTMENT_USD = 1000.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
RSI_LONG_THRESHOLD = 55.0
RSI_SHORT_THRESHOLD = 45.0
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
VOLUME_SMA_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.10

# ================= STATE =================
times = []
prices = []
volumes = []
alert_high = None
alert_low = None
last_price = None
RESOLVED_SYMBOL = None
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
})
YAHOO_COOLDOWN_UNTIL = 0.0
LAST_429_LOG_AT = 0.0
LAST_401_LOG_AT = 0.0


# ================= DATA =================
def normalize_symbol(symbol):
    s = (symbol or "").strip().upper()
    if ":" in s:
        s = s.split(":")[-1].strip()
    if s.startswith("/"):
        s = s[1:]
    return s


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _extract_price(q):
    for k in ("regularMarketPrice", "postMarketPrice", "preMarketPrice", "bid", "ask", "previousClose"):
        p = _to_float(q.get(k))
        if p is not None and p > 0:
            return p
    bid = _to_float(q.get("bid"))
    ask = _to_float(q.get("ask"))
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return None


def _extract_volume(q):
    for k in ("regularMarketVolume", "postMarketVolume", "averageDailyVolume3Month"):
        v = _to_float(q.get(k))
        if v is not None and v > 0:
            return v
    return None


class YahooRateLimited(Exception):
    pass


class YahooUnauthorized(Exception):
    pass


def _new_yahoo_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://finance.yahoo.com/",
    })
    return s


def _yahoo_get_json(url, params):
    global YAHOO_COOLDOWN_UNTIL, LAST_429_LOG_AT, LAST_401_LOG_AT, SESSION

    now = time.time()
    if now < YAHOO_COOLDOWN_UNTIL:
        raise YahooRateLimited("Yahoo cooldown active")

    for attempt in range(3):
        try:
            urls = [url]
            if "query1.finance.yahoo.com" in url:
                urls.append(url.replace("query1.finance.yahoo.com", "query2.finance.yahoo.com"))

            r = None
            for u in urls:
                r = SESSION.get(u, params=params, timeout=6)
                if r.status_code != 401:
                    break

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                #wait_s = float(retry_after) if retry_after else (120 + attempt * 60)
                wait_s = float(retry_after) if retry_after else (15+ attempt * 15)
                wait_s += random.uniform(0, 1.5)
                YAHOO_COOLDOWN_UNTIL = time.time() + wait_s

                if time.time() - LAST_429_LOG_AT > 5:
                    print(f"Yahoo rate-limited (429). Backing off for ~{int(wait_s)}s.")
                    LAST_429_LOG_AT = time.time()

                raise YahooRateLimited("429 Too Many Requests")

            if r.status_code == 401:
                # Refresh session once, then retry.
                SESSION = _new_yahoo_session()
                if attempt == 2:
                    if time.time() - LAST_401_LOG_AT > 5:
                        print("Yahoo returned 401 Unauthorized. Falling back when available.")
                        LAST_401_LOG_AT = time.time()
                    raise YahooUnauthorized("401 Unauthorized")
                time.sleep(0.6 + random.uniform(0, 0.4))
                continue

            r.raise_for_status()
            return r.json()
        except YahooRateLimited:
            raise
        except YahooUnauthorized:
            raise
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep((0.7 * (2 ** attempt)) + random.uniform(0, 0.4))


def _yfinance_quote(symbol):
    if yf is None:
        return None, None
    try:
        t = yf.Ticker(symbol)
        vol = None
        info = getattr(t, "fast_info", None)
        if info:
            for key in ("last_price", "regular_market_price", "previous_close"):
                p = _to_float(info.get(key))
                if p is not None and p > 0:
                    v = _to_float(info.get("last_volume"))
                    return p, (v if v is not None and v > 0 else None)
        hist = t.history(period="1d", interval="1m")
        if not hist.empty:
            p = _to_float(hist["Close"].dropna().iloc[-1])
            if "Volume" in hist and not hist["Volume"].dropna().empty:
                v = _to_float(hist["Volume"].dropna().iloc[-1])
                if v is not None and v > 0:
                    vol = v
            if p is not None and p > 0:
                return p, vol
    except Exception:
        return None, None
    return None, None


def _yahoo_quote(symbols):
    if isinstance(symbols, (list, tuple)):
        symbols = ",".join(symbols)
    data = _yahoo_get_json(
        "https://query1.finance.yahoo.com/v7/finance/quote",
        {"symbols": symbols}
    )
    return data.get("quoteResponse", {}).get("result", [])


def _search_yahoo_quotes(query):
    data = _yahoo_get_json(
        "https://query1.finance.yahoo.com/v1/finance/search",
        {"q": query, "quotesCount": 50, "newsCount": 0}
    )
    return data.get("quotes", [])


def _score_candidate(root, q):
    sym = (q.get("symbol") or "").upper()
    qtype = (q.get("quoteType") or "").upper()
    score = 0
    if qtype == "FUTURE":
        score += 40
    if sym == root:
        score += 80
    if sym == f"{root}=F":
        score += 100
    if sym.endswith("=F"):
        score += 30
    if sym.startswith(root):
        score += 15
    if any(x in sym for x in (".NYM", ".CME", ".CBT", ".COMEX")):
        score += 10
    return score


def _resolve_futures_symbol(user_symbol):
    root = normalize_symbol(user_symbol)
    user_root = root[:-2] if root.endswith("=F") else root

    cands = [root]
    if not root.endswith("=F"):
        cands.append(f"{root}=F")
    else:
        cands.append(root[:-2])

    # direct quote pass
    best = None
    best_s = -1
    for q in _yahoo_quote(cands):
        p = _extract_price(q)
        if p is None:
            continue
        s = _score_candidate(user_root, q)
        if s > best_s:
            best_s = s
            best = q
    if best:
        return best["symbol"], _extract_price(best)

    # search fallback
    hits = []
    for q in cands:
        hits.extend(_search_yahoo_quotes(q))
    uniq = {}
    for h in hits:
        sym = h.get("symbol")
        if sym:
            uniq[sym] = h

    ranked = sorted(uniq.values(), key=lambda x: _score_candidate(user_root, x), reverse=True)
    top = [r["symbol"] for r in ranked[:20]]
    verified = _yahoo_quote(top)

    best = None
    best_s = -1
    for q in verified:
        p = _extract_price(q)
        if p is None:
            continue
        s = _score_candidate(user_root, q)
        if s > best_s:
            best_s = s
            best = q

    if not best:
        raise ValueError(f"Could not resolve a live quote for '{user_symbol}'")
    return best["symbol"], _extract_price(best)


def fetch_price():
    global RESOLVED_SYMBOL
    s = normalize_symbol(SYMBOL)

    if "-USD" in s:
        r = requests.get(f"https://api.exchange.coinbase.com/products/{s}/ticker", timeout=6)
        r.raise_for_status()
        payload = r.json()
        p = _to_float(payload.get("price"))
        v = _to_float(payload.get("volume"))
        if p is None:
            raise ValueError(f"Coinbase ticker for {s} did not return price")
        return p, (v if v is not None and v > 0 else None)

    try:
        if RESOLVED_SYMBOL is None:
            RESOLVED_SYMBOL, p = _resolve_futures_symbol(s)
            print(f"Resolved {SYMBOL} -> {RESOLVED_SYMBOL}")
            if p is not None:
                return p, None

        q = _yahoo_quote([RESOLVED_SYMBOL])
        if q:
            p = _extract_price(q[0])
            if p is not None:
                return p, _extract_volume(q[0])

        RESOLVED_SYMBOL, p = _resolve_futures_symbol(s)
        print(f"Re-resolved {SYMBOL} -> {RESOLVED_SYMBOL}")
        if p is not None:
            return p, None
    except YahooUnauthorized:
        # Continue to yfinance fallback below.
        pass

    fallback_symbol = RESOLVED_SYMBOL or (s if s.endswith("=F") else f"{s}=F")
    p, v = _yfinance_quote(fallback_symbol)
    if p is not None:
        return p, v
    raise ValueError(
        f"No price for {SYMBOL} ({fallback_symbol}). "
        "If 401 persists, install/update yfinance: pip install -U yfinance"
    )


def compute_y_range(vals, lo, hi):
    values = vals[:]
    if lo is not None:
        values.append(lo)
    if hi is not None:
        values.append(hi)
    raw_min = min(values)
    raw_max = max(values)
    span = raw_max - raw_min
    if span == 0:
        span = abs(raw_max) * 0.01 or 1.0
    magnitude = 10 ** math.floor(math.log10(span))
    residual = span / magnitude
    if residual >= 5:
        nice = 1 * magnitude
    elif residual >= 2:
        nice = 0.5 * magnitude
    else:
        nice = 0.2 * magnitude
    step = nice / 2
    y_min = math.floor(raw_min / step) * step
    y_max = math.ceil(raw_max / step) * step
    return y_min, y_max


def _ema(arr, period):
    out = np.empty(len(arr), dtype=float)
    alpha = 2.0 / (period + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _compute_macd(arr):
    fast = _ema(arr, MACD_FAST)
    slow = _ema(arr, MACD_SLOW)
    macd = fast - slow
    signal = _ema(macd, MACD_SIGNAL)
    return macd, signal


def _compute_rsi(arr, period=RSI_PERIOD):
    rsi = np.full(len(arr), np.nan, dtype=float)
    if len(arr) <= period:
        return rsi

    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, len(arr)):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _rolling_mean_nan(values, period):
    out = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        start = max(0, i - period + 1)
        window = values[start:i + 1]
        valid = window[~np.isnan(window)]
        if valid.size > 0:
            out[i] = float(np.mean(valid))
    return out


def compute_entry_signals(close_prices, vols):
    n = len(close_prices)
    longs = np.zeros(n, dtype=bool)
    shorts = np.zeros(n, dtype=bool)
    if n < max(MACD_SLOW + MACD_SIGNAL, RSI_PERIOD + 2):
        return longs, shorts, None, None, None

    close = np.array(close_prices, dtype=float)
    vol = np.array([np.nan if v is None else float(v) for v in vols], dtype=float)
    vol_sma = _rolling_mean_nan(vol, VOLUME_SMA_PERIOD)
    macd, signal = _compute_macd(close)
    rsi = _compute_rsi(close)

    for i in range(1, n):
        cross_up = macd[i - 1] <= signal[i - 1] and macd[i] > signal[i]
        cross_down = macd[i - 1] >= signal[i - 1] and macd[i] < signal[i]
        rsi_long_ok = (not np.isnan(rsi[i])) and RSI_LONG_THRESHOLD <= rsi[i] <= RSI_OVERBOUGHT
        rsi_short_ok = (not np.isnan(rsi[i])) and RSI_OVERSOLD <= rsi[i] <= RSI_SHORT_THRESHOLD
        volume_ok = (
            not np.isnan(vol[i])
            and not np.isnan(vol_sma[i])
            and vol[i] >= (vol_sma[i] * VOLUME_SPIKE_MULTIPLIER)
        )
        longs[i] = cross_up and rsi_long_ok and volume_ok
        shorts[i] = cross_down and rsi_short_ok and volume_ok
    return longs, shorts, macd, signal, rsi


def beep():
    print("\a", end="")


# ================= PLOT =================
fig, ax = plt.subplots(figsize=(12, 6))

def on_click(event):
    global alert_low, alert_high
    if event.inaxes != ax or event.ydata is None:
        return
    tapped = float(event.ydata)
    if alert_low is None:
        alert_low = tapped
        beep()
    elif alert_high is None:
        alert_low, alert_high = min(alert_low, tapped), max(alert_low, tapped)
        beep()
    else:
        alert_low = tapped
        alert_high = None
        beep()
    redraw()

fig.canvas.mpl_connect("button_press_event", on_click)


def redraw():
    ax.clear()
    if len(times) < 2:
        fig.suptitle("")
        ax.set_title(f"{RESOLVED_SYMBOL or SYMBOL} (waiting for data...)")
        ax.grid(True, alpha=0.2)
        return

    y_min, y_max = compute_y_range(prices, alert_low, alert_high)
    if len(times) >= 4:
        x = mdates.date2num(times)
        y = np.array(prices, dtype=float)
        # Visual interpolation only; sampled points remain unchanged.
        x_dense = np.linspace(x[0], x[-1], len(x) * 4)
        y_dense = np.interp(x_dense, x, y)
        ax.plot(mdates.num2date(x_dense), y_dense, color=LINE_COLOR, linewidth=1.6)
    else:
        ax.plot(times, prices, color=LINE_COLOR, linewidth=1.6)

    long_entries, short_entries, macd, signal, rsi = compute_entry_signals(prices, volumes)
    if np.any(long_entries):
        long_idx = np.where(long_entries)[0]
        ax.scatter(
            [times[i] for i in long_idx],
            [prices[i] for i in long_idx],
            color="limegreen",
            marker="^",
            s=80,
            edgecolors="black",
            linewidths=0.4,
            label="Long entry",
            zorder=5,
        )
    if np.any(short_entries):
        short_idx = np.where(short_entries)[0]
        ax.scatter(
            [times[i] for i in short_idx],
            [prices[i] for i in short_idx],
            color="magenta",
            marker="v",
            s=80,
            edgecolors="black",
            linewidths=0.4,
            label="Short entry",
            zorder=5,
        )

    if alert_low is not None:
        ax.axhline(alert_low, color="orange", linewidth=1.0)
    if alert_high is not None:
        ax.axhline(alert_high, color="red", linewidth=1.0)

    current = prices[-1]
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{RESOLVED_SYMBOL or SYMBOL}   ${current:,.2f}")
    if normalize_symbol(SYMBOL) == "BTC-USD" and prices[0] > 0:
        start_price = prices[0]
        units = BTC_INVESTMENT_USD / start_price
        current_value = units * current
        pl = current_value - BTC_INVESTMENT_USD
        pl_pct = (pl / BTC_INVESTMENT_USD) * 100.0
        pl_color = "limegreen" if pl >= 0 else "red"
        fig.suptitle(
            f"P/L on ${BTC_INVESTMENT_USD:,.0f} BTC investment: {pl:+,.2f} USD ({pl_pct:+.2f}%)",
            color=pl_color,
            fontsize=12,
            fontweight="bold",
        )
    else:
        fig.suptitle("")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)
    if macd is not None and signal is not None and rsi is not None:
        latest_rsi = rsi[-1]
        latest_macd = macd[-1]
        latest_signal = signal[-1]
        latest_vol = volumes[-1]
        vol_text = "n/a"
        if latest_vol is not None:
            vol_text = f"{latest_vol:,.0f}"
        rsi_text = "n/a" if np.isnan(latest_rsi) else f"{latest_rsi:.1f}"
        ax.text(
            0.01,
            0.95,
            f"MACD {latest_macd:.4f} | Signal {latest_signal:.4f} | RSI {rsi_text} | Vol {vol_text}",
            transform=ax.transAxes,
            fontsize=9,
            color="white" if plt.rcParams["axes.facecolor"] != "white" else "black",
            alpha=0.85,
            va="top",
        )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    ax.text(
        0.01,
        0.01,
        "Click 1: set LOW (orange), Click 2: set HIGH (red), Click 3: restart",
        transform=ax.transAxes,
        fontsize=9,
        color="white" if plt.rcParams["axes.facecolor"] != "white" else "black",
        alpha=0.8,
    )


def tick(_frame):
    global last_price, times, prices, volumes
    try:
        now = datetime.now()
        price, volume = fetch_price()

        times.append(now)
        prices.append(price)
        volumes.append(volume)

        if last_price is not None:
            if alert_high is not None and last_price < alert_high <= price:
                beep()
            if alert_low is not None and last_price > alert_low >= price:
                beep()

        last_price = price

        cutoff = now - timedelta(seconds=WINDOW_SECONDS)
        while times and times[0] < cutoff:
            times.pop(0)
            prices.pop(0)
            volumes.pop(0)

        redraw()
    except YahooRateLimited:
        # Cooldown active: skip this cycle and try again next tick.
        return
    except YahooUnauthorized:
        # Yahoo denied this cycle; fetch_price already attempts fallback.
        return
    except Exception as e:
        print("Error:", e)


ani = FuncAnimation(fig, tick, interval=INTERVAL * 1000, cache_frame_data=False)
plt.show()