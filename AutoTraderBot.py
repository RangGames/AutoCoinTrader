#
# 면책 조항 (Disclaimer)
#
# 본 소프트웨어는 교육 및 연구 목적으로 제공되며, 어떠한 경우에도 투자 조언으로 간주되어서는 안 됩니다. 자동매매 시스템은 예기치 않은 버그나 시장의 급격한 변동으로 인해 심각한 금전적 손실을 야기할 수 있습니다.
#
# 투자에 대한 모든 책임은 사용자 본인에게 있습니다. 실제 자금으로 봇을 운영하기 전에는 반드시 충분한 모의투자와 백테스팅을 통해 그 성능과 리스크를 검증하시기 바랍니다.
#

from __future__ import annotations
import math, json, uuid, pathlib, logging, asyncio, random, re, hashlib
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import feedparser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pythonjsonlogger import jsonlogger

import ccxt.async_support as ccxt_async
try:
    import ccxt.pro as ccxtpro
    HAS_PRO = True
except Exception:
    HAS_PRO = False

import aiohttp

KST = ZoneInfo("Asia/Seoul")
BASE_DIR = pathlib.Path(".")
LOG_DIR = BASE_DIR/"logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = BASE_DIR/"reports"; REPORT_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = LOG_DIR/"state.json"

BITHUMB_API_KEY: str = "" # 필수
BITHUMB_API_SECRET: str = "" # 필수

DEEPSEEK_BASE: str = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY: str = "" # 선택
NEWSAPI_KEY: str = "" # 선택
TELEGRAM_BOT_TOKEN: str = "" # 선택
TELEGRAM_CHAT_ID: str = "" # 선택
DISCORD_WEBHOOK_URL: str = "" # 선택

# ===================== 설정 =====================
@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: ["BTC/KRW", "ETH/KRW"])  # 다중 심볼
    timeframe: str = "1d"
    k: float = 0.5
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp_rr: float = 2.0
    alloc_pct: float = 0.20
    min_notional: float = 5000.0
    fee_rate: float = 0.0004
    poll_sec: float = 3.0        # 메인 루프 주기(초)
    dry_run: bool = True
    dry_initial_equity: float = 1_000_000.0  # 드라이런 모의 시작 자산(KRW)
    # 가중치
    w_tech: float = 0.45
    w_llm: float = 0.25
    w_news: float = 0.15
    w_news_kr: float = 0.05
    w_flow: float = 0.10
    enter_threshold: float = 0.55
    # 리스크
    risk_pct: float = 0.01
    max_trades_per_day: int = 6  # 0이면 무제한
    daily_dd_stop_pct: float = 0.06
    # 웹소켓(체결 흐름)
    use_ws: bool = True
    ws_ttl_sec: int = 30
    # 주문 정책
    limit_offset_bps: float = 3.0  # 매수: bid*(1-off), 매도: ask*(1+off)
    post_only_timeout_sec: float = 2.5
    # API
    deepseek_base: str = DEEPSEEK_BASE
    deepseek_key: Optional[str] = DEEPSEEK_API_KEY
    newsapi_key: Optional[str] = NEWSAPI_KEY
    rss_feeds_kr: List[str] = field(default_factory=lambda: [
        "https://kr.cointelegraph.com/rss",
        "https://www.hankyung.com/feed/all-news",
        "https://www.yna.co.kr/rss/news.xml",
    ])
    telegram_token: Optional[str] = TELEGRAM_BOT_TOKEN
    telegram_chat_id: Optional[str] = TELEGRAM_CHAT_ID
    discord_webhook: Optional[str] = DISCORD_WEBHOOK_URL
    # 레이트리밋/뉴스 쿼터
    bithumb_rps: float = 100.0
    newsapi_daily_quota: int = 300
    news_refresh_sec: int = 1800
    # LLM 설정
    llm_broadcast: bool = True
    llm_debug: bool = True
    llm_ttl_sec: int = 300
    llm_temperature: float = 0.1
    # 예측 브로드캐스트 설정
    forecast_broadcast: bool = True
    forecast_interval_sec: int = 300

CFG = Config()

# ===================== 로깅 =====================
class Logger:
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.logger = logging.getLogger("bot"); self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(ch)
        fh = RotatingFileHandler(LOG_DIR/"bot.jsonl", maxBytes=8_000_000, backupCount=6, encoding="utf-8")
        jf = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(jf); self.logger.addHandler(fh)
        # CSV 로그 파일 생성
        self.trade_csv = LOG_DIR/"trades.csv"
        if not self.trade_csv.exists():
            pd.DataFrame(columns=[
                "session","date","time","symbol","side","price","amount","notional","reason",
                "tech","llm","news","news_kr","flow","combo","equity_before","equity_after","pnl_pct"
            ]).to_csv(self.trade_csv, index=False)
        self.equity_csv = LOG_DIR/"equity.csv"
        if not self.equity_csv.exists():
            pd.DataFrame(columns=["time","equity","peak","dd"]).to_csv(self.equity_csv, index=False)

    def log_trade(self, **row):
        row.setdefault("session", self.session_id)
        pd.DataFrame([row]).to_csv(self.trade_csv, mode="a", header=False, index=False)
        self.logger.info({"event":"trade", **row})

    def log_equity(self, equity: float, peak: float):
        dd = 0.0 if peak<=0 else (peak - equity)/peak
        pd.DataFrame([[datetime.now(KST).isoformat(), equity, peak, dd]], columns=["time","equity","peak","dd"]).to_csv(self.equity_csv, mode="a", header=False, index=False)
        self.logger.info({"event":"equity","equity":equity,"peak":peak,"dd":dd})

    def info(self, msg, **kv): self.logger.info({"session": self.session_id, "msg": msg, **kv})
    def warn(self, msg, **kv): self.logger.warning({"session": self.session_id, "msg": msg, **kv})
    def error(self, msg, **kv): self.logger.error({"session": self.session_id, "msg": msg, **kv})

LOG = Logger()

# ===================== 비동기 토큰버킷 =====================
class AsyncTokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.ts = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
            self.ts = now
            if self.tokens < 1:
                need = (1 - self.tokens) / self.rate
                await asyncio.sleep(max(need, 0))
                self.tokens = 0
            else:
                self.tokens -= 1

BUCKET = AsyncTokenBucket(CFG.bithumb_rps, CFG.bithumb_rps)

async def ratelimited_call(fn: Callable, *a, **kw):
    for i in range(5):
        try:
            await BUCKET.wait()
            return await fn(*a, **kw)
        except ccxt_async.NetworkError as e:
            sleep = min(2 ** i + random.random(), 8)
            LOG.warn("네트워크_재시도", error=str(e), retry_in=sleep)
            await asyncio.sleep(sleep)
        except ccxt_async.ExchangeError as e:
            if "rate" in str(e).lower():
                await asyncio.sleep(1)
                continue
            raise

# ===================== 캐시 매니저(전역 상태) =====================
class CacheManager:
    def __init__(self):
        self.llm: Dict[str, Tuple[float, float]] = {}
        self.news: Dict[str, Tuple[float, float]] = {}
        self.flow: Dict[str, List[Tuple[float, float]]] = {}
        self.notice: Tuple[float, List[Any]] = (0.0, [])
        self.news_calls_today = 0
        self.news_calls_day = datetime.now(KST).strftime('%Y-%m-%d')
        self.lock = asyncio.Lock()

    def clamp01(self, x: float) -> float: return max(0.0, min(1.0, float(x)))

CACHE = CacheManager()

# ===================== 알림(비동기) =====================
async def notify_text(session:aiohttp.ClientSession, msg:str):
    tasks=[]
    if CFG.telegram_token and CFG.telegram_chat_id:
        url=f"https://api.telegram.org/bot{CFG.telegram_token}/sendMessage"
        tasks.append(session.post(url,json={"chat_id":CFG.telegram_chat_id,"text":msg[:3900]}))
    if CFG.discord_webhook:
        tasks.append(session.post(CFG.discord_webhook,json={"content":msg[:1900]}))
    if tasks:
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass

async def notify_trade(session:aiohttp.ClientSession, row:Dict):
    reasoning = (
        f"점수 → 기술:{row.get('tech',0):.2f} | LLM:{row.get('llm',0):.2f} | 영문뉴스:{row.get('news',0):.2f} | 국문뉴스:{row.get('news_kr',0):.2f} | 체결흐름:{row.get('flow',0):.2f} | 종합:{row.get('combo',0):.2f}"
        f"위험 → 손익%:{100*row.get('pnl_pct',0):.2f}% | 전자산:{row.get('equity_before',0):.0f} → 후자산:{row.get('equity_after',0):.0f}"
        f"사유 → {row.get('reason','')} (목표가 돌파 및 종합점수 기준 충족, 손절/익절 규칙)"
    )
    txt = f"[{row.get('symbol')}] {row.get('side')} {row.get('amount',0):.6f} @ {row.get('price',0):.0f} " + reasoning
    await notify_text(session, txt)

# ===================== HTTP 도우미 =====================
async def http_get_json(session:aiohttp.ClientSession, url:str, **params):
    async with session.get(url, params=params.get('params'), headers=params.get('headers'), timeout=params.get('timeout',10)) as r:
        return await r.json(content_type=None)

async def http_post_json(session:aiohttp.ClientSession, url:str, json_body:dict, headers:dict=None, timeout:int=10):
    async with session.post(url, json=json_body, headers=headers, timeout=timeout) as r:
        return await r.json(content_type=None)

# ===================== 데이터/지표 =====================
async def fetch_ohlcv_df(ex: ccxt_async.Exchange, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    ohlcv = await ratelimited_call(ex.fetch_ohlcv, symbol, timeframe, None, limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(KST)
    return df

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr

def build_signal_state(df: pd.DataFrame, k: float, atr_period: int):
    df = df.copy()
    df["atr"] = calc_atr(df, atr_period)
    df["range"] = df["high"].shift(1) - df["low"].shift(1)
    df["target"] = df["open"] + k * df["range"]
    return df

# ===================== 스코어링 =====================
async def tech_score_today(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 0.0
    row, prev = df.iloc[-1], df.iloc[-2]
    atr = float(row["atr"]) if not math.isnan(row["atr"]) else 0.0
    if atr <= 0: return 0.0
    target = float(row["target"])
    high, open_, close, vol = map(float, [row["high"], row["open"], row["close"], row["volume"]])
    prev_vol = float(prev["volume"]) if prev["volume"]>0 else vol
    breakout = max(0.0, min(1.0, (high - target) / (atr * 1.0))) if high > target else 0.0
    body = max(0.0, min(1.0, (close - open_) / atr))
    vol_boost = max(0.0, min(1.0, (vol - prev_vol) / max(prev_vol, 1e-9)))
    return 0.6*breakout + 0.3*body + 0.1*vol_boost

async def llm_score(session:aiohttp.ClientSession, symbol: str, df: pd.DataFrame, ttl: float = None) -> float:
    now = asyncio.get_event_loop().time()
    if ttl is None:
        ttl = CFG.llm_ttl_sec
    if symbol in CACHE.llm and now - CACHE.llm[symbol][1] < ttl:
        LOG.info("LLM_캐시적중", symbol=symbol)
        return CACHE.llm[symbol][0]
    if CFG.w_llm <= 0 or not CFG.deepseek_key:
        CACHE.llm[symbol] = (0.0, now); return 0.0
    tail = df.tail(24)
    last = tail[["open","high","low","close","volume"]].to_dict(orient="records")
    try:
        start_ts = str(tail["ts"].iloc[0])
        end_ts = str(tail["ts"].iloc[-1])
    except Exception:
        start_ts = end_ts = ""
    inp_hash = hashlib.sha256(json.dumps(last, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    LOG.info("LLM_입력요약", symbol=symbol, candles=len(last), window=f"{start_ts}..{end_ts}", digest=inp_hash)
    sys_msg = {"role":"system","content":(
        "You are a quantitative trading assistant."
        "Respond ONLY with a compact JSON object with keys trend, breakout, risk."
        "Each value must be a number between 0 and 1. No prose, no markdown.")}
    # 심볼 차별화를 위한 통계 피처 추가
    closes = [float(x.get('close', 0.0)) for x in last if isinstance(x, dict)]
    ret7 = 0.0; ret24 = 0.0; vol24 = 0.0
    if len(closes) >= 2:
        rets = []
        for i in range(1, len(closes)):
            prev = closes[i-1] or 0.0
            cur = closes[i] or 0.0
            rets.append(0.0 if prev == 0 else (cur/prev - 1.0))
        if rets:
            vol24 = float(pd.Series(rets).std()) if hasattr(pd, 'Series') else float(sum((r - (sum(rets)/len(rets)))**2 for r in rets)/len(rets))**0.5
    if len(closes) >= 7:
        ret7 = 0.0 if closes[-7] == 0 else (closes[-1]/closes[-7] - 1.0)
    if len(closes) >= 24:
        ret24 = 0.0 if closes[-24] == 0 else (closes[-1]/closes[-24] - 1.0)
    try:
        atr_last = float(tail['atr'].iloc[-1]) if 'atr' in tail.columns else 0.0
        close_last = float(tail['close'].iloc[-1])
        atr_pct = 0.0 if close_last == 0 else (atr_last/close_last)
    except Exception:
        atr_last = 0.0; atr_pct = 0.0
    features = {"ret7":ret7, "ret24":ret24, "vol24":vol24, "atr":atr_last, "atr_pct":atr_pct}
    user_msg = {"role":"user","content":(
        f"Symbol: {symbol}. Given the last 24 daily candles (OHLCV) and the summary features below, estimate next 24h:"
        " {trend, breakout, risk}."
        f" candles={json.dumps(last)} features={json.dumps(features)}")}

    def _extract_json(text: str) -> dict:
        text = text.strip()
        if text.startswith('{') and text.endswith('}'):
            try:
                return json.loads(text)
            except Exception:
                pass
        # 코드블록 또는 앞뒤 텍스트 포함 시 첫 중괄호부터 밸런싱해 추출
        start = text.find('{')
        if start == -1:
            return {}
        depth = 0
        for i in range(start, min(len(text), start + 5000)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    fragment = text[start:i+1]
                    try:
                        return json.loads(fragment)
                    except Exception:
                        break
        # 키워드 기반 수치 추출(폴백)
        out = {}
        for key in ("trend","breakout","risk"):
            m = re.search(rf"{key}[^0-9]*([01](?:\\.[0-9]+)?)", text, flags=re.I)
            if m:
                try:
                    out[key] = float(m.group(1))
                except Exception:
                    pass
        return out
    try:
        req_body = {"model":"deepseek-chat","messages":[sys_msg, user_msg],"temperature":CFG.llm_temperature,"max_tokens":150}
        # 일부 API는 response_format을 지원하지 않을 수 있으므로 옵션으로만 포함
        req_body["response_format"] = {"type":"json_object"}
        j = await http_post_json(session, f"{CFG.deepseek_base}/chat/completions", json_body=req_body,
                                 headers={"Authorization":f"Bearer {CFG.deepseek_key}","Content-Type":"application/json"})
        if "error" in j:
            LOG.warn("LLM_API_오류", detail=str(j.get("error")))
            raise RuntimeError("LLM API error")
        text = (j.get("choices", [{}])[0].get("message", {}) or {}).get("content", "{}")
        if CFG.llm_debug:
            LOG.info("LLM_원문", symbol=symbol, preview=text[:300])
        parsed = _extract_json(text)
        trend = CACHE.clamp01(parsed.get("trend", 0.5))
        breakout = CACHE.clamp01(parsed.get("breakout", 0.5))
        risk = CACHE.clamp01(parsed.get("risk", 0.5))
        score = CACHE.clamp01(0.6*breakout + 0.3*trend + 0.1*(1-risk))
        # LLM 결과 로깅 및 알림(옵션)
        LOG.info("LLM_결과", symbol=symbol, trend=trend, breakout=breakout, risk=risk, score=score)
        if CFG.llm_broadcast:
            try:
                msg = (
                    f"[LLM] {symbol} 결과\n"
                    f"- 추세: {trend:.2f}\n- 돌파: {breakout:.2f}\n- 위험: {risk:.2f}\n- 종합점수: {score:.2f}"
                )
                await notify_text(session, msg)
            except Exception:
                pass
    except Exception as e:
        LOG.warn("LLM_예외", error=str(e))
        trend = breakout = risk = 0.5
        score = 0.0
        if CFG.llm_broadcast:
            try:
                await notify_text(session, f"[LLM] {symbol} 오류: {e}")
            except Exception:
                pass
    CACHE.llm[symbol] = (score, now); return score

async def _news_quota_ok() -> bool:
    today = datetime.now(KST).strftime('%Y-%m-%d')
    if today != CACHE.news_calls_day:
        CACHE.news_calls_day = today
        CACHE.news_calls_today = 0
    return CACHE.news_calls_today < int(CFG.newsapi_daily_quota*0.9)

async def news_score_en(session:aiohttp.ClientSession, ttl: float = None) -> float:
    if ttl is None: ttl = CFG.news_refresh_sec
    now = asyncio.get_event_loop().time(); key = "EN"
    if key in CACHE.news and now - CACHE.news[key][1] < ttl:
        return CACHE.news[key][0]
    if CFG.w_news <= 0 or not CFG.newsapi_key or not (await _news_quota_ok()):
        CACHE.news[key] = (0.0, now); return 0.0
    try:
        j = await http_get_json(session, "https://newsapi.org/v2/everything",
                                params={"q":"(bitcoin OR ethereum) AND (volatility OR breakout OR regulation)","language":"en","sortBy":"publishedAt","pageSize":20},
                                headers={"Authorization":CFG.newsapi_key})
        CACHE.news_calls_today += 1
        arts = j.get("articles", [])
        pos = ["ETF","approval","adoption","institutional","bull","breakout"]; neg = ["ban","hack","exploit","fraud","lawsuit","downturn","liquidation"]
        s=0
        for a in arts[:10]:
            t = ((a.get("title") or "") + " " + (a.get("description") or "")).lower()
            if any(p.lower() in t for p in pos): s+=1
            if any(n.lower() in t for n in neg): s-=1
        score = CACHE.clamp01(0.5 + 0.07*s)
    except Exception:
        score = 0.0
    CACHE.news[key]=(score, now); return score

async def news_score_kr(ttl: float = None) -> float:
    if ttl is None: ttl = CFG.news_refresh_sec
    now = asyncio.get_event_loop().time(); key = "KR"
    if key in CACHE.news and now - CACHE.news[key][1] < ttl:
        return CACHE.news[key][0]
    try:
        pos=["승인","상승","채택","호재","호황","확대"]; neg=["하락","해킹","사기","규제","소송","청산"]
        s=0
        for url in CFG.rss_feeds_kr:
            feed = feedparser.parse(url)
            for e in (feed.entries or [])[:8]:
                t=(getattr(e,"title","") or "") + " " + (getattr(e,"summary","") or "")
                if any(p in t for p in pos): s+=1
                if any(n in t for n in neg): s-=1
        score = CACHE.clamp01(0.5 + 0.05*s)
    except Exception:
        score = 0.0
    CACHE.news[key]=(score, now); return score

# ===== 공지(점검/유의) =====
async def should_pause_trading_from_notices(session:aiohttp.ClientSession) -> bool:
    now = asyncio.get_event_loop().time()
    last, cache = CACHE.notice
    if now - last < 60:
        text = " ".join([str(x) for x in cache])
        return any(k in text for k in ["점검","지갑","일시중단","유의종목","주의"])
    try:
        j = await http_get_json(session, "https://api.bithumb.com/v1/notices", params={"count":5}, timeout=7)
        CACHE.notice = (now, j if isinstance(j,list) else j.get("data", j))
    except Exception:
        pass
    text = " ".join([str(x) for x in CACHE.notice[1]])
    return any(k in text for k in ["점검","지갑","일시중단","유의종목","주의"])

# ===== flow score (WS) =====
async def ws_public_trades(symbols: List[str]):
    if not (CFG.use_ws and HAS_PRO):
        return
    ex = ccxtpro.bithumb({"enableRateLimit": True})
    try:
        while True:
            now = asyncio.get_event_loop().time()
            for s in symbols:
                try:
                    trades = await ex.watch_trades(s)
                    if s not in CACHE.flow: CACHE.flow[s]=[]
                    for tr in trades:
                        qty = float(tr.get("amount") or 0.0)
                        side = tr.get("side") or ""
                        signed = qty if side == "buy" else (-qty if side == "sell" else 0.0)
                        CACHE.flow[s].append((now, signed))
                    cutoff = now - CFG.ws_ttl_sec
                    CACHE.flow[s] = [(t,q) for (t,q) in CACHE.flow[s] if t >= cutoff]
                except Exception:
                    await asyncio.sleep(0.1)
            await asyncio.sleep(0.05)
    finally:
        await ex.close()

def flow_score(symbol: str) -> float:
    window = CACHE.flow.get(symbol, [])
    if not window: return 0.0
    pos = sum(max(q,0.0) for _,q in window)
    neg = sum(-min(q,0.0) for _,q in window)
    total = pos + neg
    if total <= 0: return 0.0
    buy_ratio = pos / total
    intensity = max(0.0, min(1.0, len(window)/200))
    return max(0.0, min(1.0, 0.5 + (buy_ratio-0.5)*intensity*2))

async def combined_score(session:aiohttp.ClientSession, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
    t = await tech_score_today(df)
    l = await llm_score(session, symbol, df)
    n_en = await news_score_en(session)
    n_kr = await news_score_kr()
    f = flow_score(symbol)
    combo = max(0.0, min(1.0, CFG.w_tech*t + CFG.w_llm*l + CFG.w_news*n_en + CFG.w_news_kr*n_kr + CFG.w_flow*f))
    return {"tech": t, "llm": l, "news": n_en, "news_kr": n_kr, "flow": f, "combo": combo}

# ===================== 정밀도/호가단위 =====================
async def round_amount(ex:ccxt_async.Exchange, symbol:str, amt:float)->float:
    try: return float(ex.amount_to_precision(symbol, amt))
    except Exception: return float(f"{amt:.6f}")

async def round_price(ex:ccxt_async.Exchange, symbol:str, px:float)->float:
    try: return float(ex.price_to_precision(symbol, px))
    except Exception: return float(f"{px:.0f}")

# ===================== 주문 (메이커 시도→IOC 폴백, 부분체결 차감) =====================
async def get_free_krw_and_base(ex: ccxt_async.Exchange, symbol: str) -> Tuple[float, float]:
    """현재 가용 KRW와 심볼의 베이스 코인 가용 수량을 반환.

    - 실거래: 프라이빗 잔고에서 조회
    - 모의거래: 제한 없음(inf) 반환
    """
    if CFG.dry_run:
        return float('inf'), float('inf')
    try:
        bal = await ratelimited_call(ex.fetch_balance)
        free = (bal.get('free') or {})
        krw_free = float(free.get('KRW') or 0.0)
        base = symbol.split('/')[0]
        base_free = float(free.get(base) or 0.0)
        LOG.info("잔고_스냅샷", krw_free=krw_free, base=base, base_free=base_free)
        return krw_free, base_free
    except Exception as e:
        LOG.warn("잔고조회_실패", error=str(e))
        return 0.0, 0.0
async def limit_price_from_book(book:Dict, side:str, offset_bps:float)->float:
    bid = book['bids'][0][0] if book['bids'] else 0.0
    ask = book['asks'][0][0] if book['asks'] else 0.0
    return bid*(1.0 - offset_bps/10000.0) if side=="buy" else ask*(1.0 + offset_bps/10000.0)

async def place_post_only_with_ioc_fallback(ex:ccxt_async.Exchange, symbol:str, side:str, amount:float)->Dict:
    if CFG.dry_run:
        LOG.info("모의주문", symbol=symbol, side=side, amount=amount)
        return {"id":"dry","status":"closed"}
    remaining = await round_amount(ex, symbol, amount)
    try:
        book = await ratelimited_call(ex.fetch_order_book, symbol)
        px = await round_price(ex, symbol, await limit_price_from_book(book, side, CFG.limit_offset_bps))
        LOG.info("주문_시도", symbol=symbol, side=side, type="limit", amount=remaining, price=px)
        order = await ratelimited_call(ex.create_order, symbol, "limit", side, remaining, px, {})
        t0=asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time()-t0 < CFG.post_only_timeout_sec:
            try:
                o = await ratelimited_call(ex.fetch_order, order['id'], symbol)
                status = (o.get('status') or '').lower(); filled = float(o.get('filled') or 0.0)
                remaining = max(0.0, remaining - filled)
                if status in ("closed","canceled") or remaining <= 1e-12:
                    return o
            except Exception:
                pass
            await asyncio.sleep(0.25)
        try:
            o = await ratelimited_call(ex.fetch_order, order['id'], symbol)
            filled = float(o.get('filled') or 0.0)
            remaining = max(0.0, remaining - filled)
            await ratelimited_call(ex.cancel_order, order['id'], symbol)
            LOG.info("주문_취소", symbol=symbol, order_id=order.get('id'))
        except Exception:
            pass
    except Exception as e:
        LOG.warn("메이커_시도_실패", error=str(e))
    if remaining <= 1e-12:
        return {"status":"closed","id":"maker-only-filled"}
    try:
        remaining = await round_amount(ex, symbol, remaining)
        LOG.info("IOC_주문", symbol=symbol, side=side, amount=remaining)
        return await ratelimited_call(ex.create_order, symbol, "market", side, remaining, None, {"timeInForce":"IOC"})
    except Exception as e:
        LOG.error("IOC_폴백_실패", error=str(e)); raise

# ===================== 계정/리스크 =====================
async def account_equity(ex: ccxt_async.Exchange, positions: Optional[Dict[str, Dict[str, float]]] = None) -> float:
    """계정 자산(KRW) 조회.

    - 실거래(dry_run=False): 비트햄 프라이빗 API로 잔고 조회
    - 모의거래(dry_run=True): 설정된 시작 자산 + 보유 포지션의 추정 평가손익을 반영
    """
    if CFG.dry_run:
        eq = CFG.dry_initial_equity
        if positions:
            for s, pos in positions.items():
                size = float(pos.get("size") or 0.0)
                entry = float(pos.get("entry") or 0.0)
                if size > 0 and entry > 0:
                    try:
                        tick = await ratelimited_call(ex.fetch_ticker, s)
                        price = float(tick.get("last") or tick.get("close") or 0.0)
                        eq += size * (price - entry)
                    except Exception:
                        pass
        return eq

    bal = await ratelimited_call(ex.fetch_balance)
    total = float((bal.get("total") or {}).get("KRW", 0.0))
    if total>0: return total
    krw = float((bal.get("free") or {}).get("KRW", 0.0))
    eq = krw
    for s in CFG.symbols:
        base = s.split("/")[0]
        amt = float((bal.get("free") or {}).get(base, 0.0))
        if amt>0:
            tick = await ratelimited_call(ex.fetch_ticker, s)
            px = float(tick.get("last") or tick.get("close") or 0.0)
            eq += amt*px
    return eq

async def atr_position_size(eq_krw: float, atr: float, price: float) -> float:
    risk_krw = max(0.0, eq_krw * CFG.risk_pct)
    if atr <= 0:
        budget = eq_krw * CFG.alloc_pct
        return 0.0 if budget < CFG.min_notional else budget / price
    stop_dist = atr * CFG.sl_atr_mult
    if stop_dist <= 0: return 0.0
    qty = risk_krw / stop_dist
    if qty*price < CFG.min_notional: return 0.0
    return qty

# ===================== 봇 (비동기) =====================
class Bot:
    def __init__(self, ex: ccxt_async.Exchange):
        self.ex = ex
        self.positions: Dict[str, Dict[str, float]] = {s: {"size":0.0,"entry":0.0,"stop":0.0,"take":0.0} for s in CFG.symbols}
        self.running = True
        self.trade_count_today = 0
        self.today: date = datetime.now(KST).date()
        self.day_equity_open = 0.0
        self.day_equity_peak = 0.0
        self.dd_tripped = False
        self._last_forecast_ts: float = 0.0

    # ----- 상태 저장/복구 -----
    async def save_state(self):
        try:
            data = {
                "today": str(self.today),
                "trade_count_today": self.trade_count_today,
                "positions": self.positions,
                "day_equity_open": self.day_equity_open,
                "day_equity_peak": self.day_equity_peak,
                "dd_tripped": self.dd_tripped,
            }
            tmp = STATE_FILE.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            tmp.replace(STATE_FILE)
        except Exception as e:
            LOG.warn("상태저장_실패", error=str(e))

    async def load_state(self):
        try:
            if STATE_FILE.exists():
                data = json.loads(STATE_FILE.read_text())
                if data.get("today") == str(datetime.now(KST).date()):
                    self.trade_count_today = int(data.get("trade_count_today", 0))
                    self.positions = data.get("positions", self.positions)
                    self.day_equity_open = float(data.get("day_equity_open", 0.0))
                    self.day_equity_peak = float(data.get("day_equity_peak", 0.0))
                    self.dd_tripped = bool(data.get("dd_tripped", False))
        except Exception as e:
            LOG.warn("상태복구_실패", error=str(e))

    # ----- 라이프사이클 -----
    async def on_new_day(self, session:aiohttp.ClientSession):
        self.today = datetime.now(KST).date()
        eq = await account_equity(self.ex, self.positions)
        self.day_equity_open = eq
        self.day_equity_peak = eq
        self.trade_count_today = 0
        self.dd_tripped = False
        LOG.info("새로운_일자", date=str(self.today), equity_open=eq)
        LOG.log_equity(eq, self.day_equity_peak)
        await self.save_state()
        await notify_text(session, f"[시작] {self.today} 자산={eq:,.0f} KRW")

    async def check_rollover(self, session:aiohttp.ClientSession):
        now = datetime.now(KST)
        if now.date() != self.today:
            for s, pos in self.positions.items():
                if pos["size"] > 0:
                    eq_before = await account_equity(self.ex, self.positions)
                    _, base_free = await get_free_krw_and_base(self.ex, s)
                    sell_qty = await round_amount(self.ex, s, min(pos["size"], base_free))
                    if sell_qty <= 0:
                        LOG.warn("가용코인부족_롤오버불가", symbol=s)
                        continue
                    await place_post_only_with_ioc_fallback(self.ex, s, "sell", sell_qty)
                    price = float((await ratelimited_call(self.ex.fetch_ticker, s)).get("last") or 0.0)
                    eq_after = await account_equity(self.ex, self.positions)
                    pnl = (eq_after - eq_before)/max(eq_before,1e-9)
                    LOG.log_trade(date=str(self.today), time=str(now.time()), symbol=s, side="SELL", price=price,
                                  amount=0.0, notional=0.0, reason="롤오버",
                                  tech=0, llm=0, news=0, news_kr=0, flow=0, combo=0,
                                  equity_before=eq_before, equity_after=eq_after, pnl_pct=pnl)
                    self.positions[s] = {"size":0.0,"entry":0.0,"stop":0.0,"take":0.0}
                    await self.save_state()
            await self.on_new_day(session)

    async def risk_guards(self):
        eq = await account_equity(self.ex, self.positions)
        if eq > self.day_equity_peak: self.day_equity_peak = eq
        dd = 0.0 if self.day_equity_peak<=0 else (self.day_equity_peak - eq) / self.day_equity_peak
        if dd >= CFG.daily_dd_stop_pct:
            if not self.dd_tripped: LOG.warn("일중_손실중지", dd=dd)
            self.dd_tripped = True
        LOG.log_equity(eq, self.day_equity_peak)
        await self.save_state()
        return eq

    async def run(self):
        async with aiohttp.ClientSession() as session:
            await self.load_state()
            # REST 거래소
            ex = self.ex
            try:
                await ex.load_markets()
            except Exception as e:
                LOG.warn("마켓로딩_실패", error=str(e))
            # WS 퍼블릭 트레이드
            ws_task = None
            if CFG.use_ws and HAS_PRO:
                ws_task = asyncio.create_task(ws_public_trades(CFG.symbols))
            await self.on_new_day(session)

            while self.running:
                try:
                    await self.check_rollover(session)
                    eq = await self.risk_guards()
                    now = datetime.now(KST)
                    now_ts = asyncio.get_event_loop().time()
                    can_enter = (not self.dd_tripped) and ((CFG.max_trades_per_day==0) or (self.trade_count_today < CFG.max_trades_per_day)) and (not await should_pause_trading_from_notices(session))

                    for symbol in CFG.symbols:
                        df = await fetch_ohlcv_df(ex, symbol, CFG.timeframe)
                        df = build_signal_state(df, CFG.k, CFG.atr_period)
                        row = df.iloc[-1]
                        target = float(row["target"]) ; atr = float(row["atr"]) if not math.isnan(row["atr"]) else 0.0
                        tick = await ratelimited_call(ex.fetch_ticker, symbol)
                        price = float(tick.get("last") or tick.get("close") or 0.0)

                        scores = await combined_score(session, symbol, df)
                        LOG.info("점수", symbol=symbol, **scores)

                        pos = self.positions[symbol]

                        # 진입
                        if can_enter and pos["size"] <= 0 and price >= target and scores["combo"] >= CFG.enter_threshold and atr > 0:
                            qty = await atr_position_size(eq, atr, price)
                            qty = await round_amount(ex, symbol, qty)
                            # 가용 KRW로 캡(실거래)
                            if qty > 0:
                                krw_free, _ = await get_free_krw_and_base(ex, symbol)
                                if krw_free != float('inf'):
                                    max_qty_by_cash = max(0.0, krw_free / (price * (1.0 + CFG.fee_rate)))
                                    qty = await round_amount(ex, symbol, min(qty, max_qty_by_cash))
                            if qty > 0 and qty*price >= CFG.min_notional:
                                eq_before = eq
                                await place_post_only_with_ioc_fallback(ex, symbol, "buy", qty)
                                pos.update({"size": qty, "entry": price, "stop": max(0.0, price - CFG.sl_atr_mult*atr), "take": price + (CFG.sl_atr_mult*atr)*CFG.tp_rr})
                                self.trade_count_today += 1
                                eq_after = await account_equity(ex, self.positions)
                                pnl = (eq_after - eq_before)/max(eq_before,1e-9)
                                rowlog = dict(date=str(now.date()), time=str(now.time()), symbol=symbol, side="BUY",
                                              price=price, amount=qty, notional=price*qty, reason="돌파진입",
                                              tech=scores["tech"], llm=scores["llm"], news=scores["news"], news_kr=scores["news_kr"], flow=scores["flow"], combo=scores["combo"],
                                              equity_before=eq_before, equity_after=eq_after, pnl_pct=pnl)
                                LOG.log_trade(**rowlog)
                                await notify_trade(session, rowlog)
                                await self.save_state()
                            else:
                                LOG.warn("수량_너무작음_또는_최소금액_미달", symbol=symbol)

                        # 포지션 관리/청산
                        if pos["size"] > 0:
                            if price <= pos["stop"]:
                                eq_before = await account_equity(ex, self.positions)
                                _, base_free = await get_free_krw_and_base(ex, symbol)
                                sell_qty = await round_amount(ex, symbol, min(pos["size"], base_free))
                                if sell_qty <= 0:
                                    LOG.warn("가용코인부족_손절불가", symbol=symbol)
                                    continue
                                await place_post_only_with_ioc_fallback(ex, symbol, "sell", sell_qty)
                                self.trade_count_today += 1
                                pos.update({"size":0.0,"entry":0.0,"stop":0.0,"take":0.0})
                                eq_after = await account_equity(ex, self.positions)
                                pnl = (eq_after - eq_before)/max(eq_before,1e-9)
                                rowlog = dict(date=str(now.date()), time=str(now.time()), symbol=symbol, side="SELL",
                                              price=price, amount=0.0, notional=0.0, reason="손절",
                                              tech=scores["tech"], llm=scores["llm"], news=scores["news"], news_kr=scores["news_kr"], flow=scores["flow"], combo=scores["combo"],
                                              equity_before=eq_before, equity_after=eq_after, pnl_pct=pnl)
                                LOG.log_trade(**rowlog)
                                await notify_trade(session, rowlog)
                                await self.save_state()
                            elif price >= pos["take"]:
                                eq_before = await account_equity(ex, self.positions)
                                _, base_free = await get_free_krw_and_base(ex, symbol)
                                sell_qty = await round_amount(ex, symbol, min(pos["size"], base_free))
                                if sell_qty <= 0:
                                    LOG.warn("가용코인부족_익절불가", symbol=symbol)
                                    continue
                                await place_post_only_with_ioc_fallback(ex, symbol, "sell", sell_qty)
                                self.trade_count_today += 1
                                pos.update({"size":0.0,"entry":0.0,"stop":0.0,"take":0.0})
                                eq_after = await account_equity(ex, self.positions)
                                pnl = (eq_after - eq_before)/max(eq_before,1e-9)
                                rowlog = dict(date=str(now.date()), time=str(now.time()), symbol=symbol, side="SELL",
                                              price=price, amount=0.0, notional=0.0, reason="익절",
                                              tech=scores["tech"], llm=scores["llm"], news=scores["news"], news_kr=scores["news_kr"], flow=scores["flow"], combo=scores["combo"],
                                              equity_before=eq_before, equity_after=eq_after, pnl_pct=pnl)
                                LOG.log_trade(**rowlog)
                                await notify_trade(session, rowlog)
                                await self.save_state()
                            else:
                                cutoff = now.replace(hour=23, minute=59, second=30, microsecond=0).time()
                                if now.time() >= cutoff:
                                    eq_before = await account_equity(ex, self.positions)
                                    _, base_free = await get_free_krw_and_base(ex, symbol)
                                    sell_qty = await round_amount(ex, symbol, min(pos["size"], base_free))
                                    if sell_qty <= 0:
                                        LOG.warn("가용코인부족_당일청산불가", symbol=symbol)
                                        continue
                                    await place_post_only_with_ioc_fallback(ex, symbol, "sell", sell_qty)
                                    self.trade_count_today += 1
                                    pos.update({"size":0.0,"entry":0.0,"stop":0.0,"take":0.0})
                                    eq_after = await account_equity(ex, self.positions)
                                    pnl = (eq_after - eq_before)/max(eq_before,1e-9)
                                    rowlog = dict(date=str(now.date()), time=str(now.time()), symbol=symbol, side="SELL",
                                                  price=price, amount=0.0, notional=0.0, reason="당일청산",
                                                  tech=scores["tech"], llm=scores["llm"], news=scores["news"], news_kr=scores["news_kr"], flow=scores["flow"], combo=scores["combo"],
                                                  equity_before=eq_before, equity_after=eq_after, pnl_pct=pnl)
                                    LOG.log_trade(**rowlog)
                                    await notify_trade(session, rowlog)
                                    await self.save_state()

                    # n분마다 예측 가격 브로드캐스트
                    if CFG.forecast_broadcast and (now_ts - self._last_forecast_ts >= CFG.forecast_interval_sec):
                        try:
                            msgs = []
                            for symbol in CFG.symbols:
                                df = await fetch_ohlcv_df(ex, symbol, CFG.timeframe)
                                df = build_signal_state(df, CFG.k, CFG.atr_period)
                                row = df.iloc[-1]
                                target = float(row["target"]) ; atr = float(row["atr"]) if not math.isnan(row["atr"]) else 0.0
                                tick = await ratelimited_call(ex.fetch_ticker, symbol)
                                price = float(tick.get("last") or tick.get("close") or 0.0)
                                stop = max(0.0, target - CFG.sl_atr_mult*atr)
                                take = target + (CFG.sl_atr_mult*atr)*CFG.tp_rr
                                pos = self.positions.get(symbol, {"size":0.0,"entry":0.0,"stop":0.0,"take":0.0})
                                line = (
                                    f"[{symbol}] 현재:{price:,.0f} 목표(예상 매수):{target:,.0f} "
                                    f"손절 후보:{stop:,.0f} 익절 후보:{take:,.0f} 보유수량:{pos['size']:.6f}"
                                )
                                msgs.append(line)
                            if msgs:
                                await notify_text(session, "예상 가격 업데이트\n" + "\n".join(msgs))
                            self._last_forecast_ts = now_ts
                        except Exception as e:
                            LOG.warn("브로드캐스트_실패", error=str(e))

                    await asyncio.sleep(CFG.poll_sec)
                except Exception as e:
                    LOG.error("루프_오류", error=f"{type(e).__name__}: {e}")
                    await asyncio.sleep(2)
            if ws_task: ws_task.cancel()

# ===================== 백테스트(간단) =====================
async def backtest_symbol(ex:ccxt_async.Exchange, symbol:str, start:str, end:str, k:float, atr_mult:float, atr_p:int)->Dict:
    ohlcv = await ratelimited_call(ex.fetch_ohlcv, symbol, CFG.timeframe, None, 1500)
    df=pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True).dt.tz_convert(KST)
    df=df[(df['ts']>=pd.Timestamp(start).tz_localize(KST))&(df['ts']<=pd.Timestamp(end).tz_localize(KST))].reset_index(drop=True)
    df=build_signal_state(df,k,atr_p)
    cash=1.0; pos=0.0; entry=0.0; curve=[]
    for i in range(1,len(df)):
        o,h,l,c,atr,target = [df.at[i,x] for x in ["open","high","low","close","atr","target"]]
        if pos==0 and h>=target and atr>0:
            entry=target; pos=cash/entry; cash=0.0
        if pos>0:
            stop=entry-atr*atr_mult; take=entry+(entry-stop)*CFG.tp_rr; exit_price=None
            if l<=stop: exit_price=stop
            elif h>=take: exit_price=take
            elif i==len(df)-1: exit_price=c
            if exit_price is not None:
                cash=pos*exit_price; pos=0.0
        eq=cash+(pos*c); curve.append(eq)
    ret=curve[-1]-1 if curve else 0
    return {"symbol":symbol,"k":k,"atr_mult":atr_mult,"ret":ret,"curve":curve}

async def backtest_grid(ex:ccxt_async.Exchange, symbol:str, start:str, end:str, k_list:List[float], atr_list:List[float]):
    results=[]
    for k in k_list:
        for am in atr_list:
            results.append(await backtest_symbol(ex,symbol,start,end,k,am,CFG.atr_period))
    df=pd.DataFrame([{"k":r["k"],"atr_mult":r["atr_mult"],"ret":r["ret"]} for r in results])
    pivot=df.pivot(index="k",columns="atr_mult",values="ret").sort_index()
    fig,ax=plt.subplots(figsize=(6,4)); im=ax.imshow(pivot.values,aspect='auto'); ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns]); ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([f"{i:.2f}" for i in pivot.index]); ax.set_title(f"백테스트 그리드 {symbol}"); plt.colorbar(im,ax=ax); fig.tight_layout()
    out_png=REPORT_DIR/f"bt_grid_{symbol.replace('/','_')}.png"; fig.savefig(out_png)
    pivot.to_csv(REPORT_DIR/f"bt_grid_{symbol.replace('/','_')}.csv")
    disclaimer=(
        "주의: 이 백테스트는 일봉(1D) 데이터 기반의 단순 체결 모델입니다."
        "일중 변동성, 슬리피지, 손절/익절 체결 우선순위 등은 정확히 반영되지 않아"
        "실제 성과와 차이가 발생할 수 있습니다. 더 정밀한 검증은 분봉/틱 기반 이벤트 드리븐 백테스터를 사용하세요."
    )
    (REPORT_DIR/"README_BACKTEST.txt").write_text(disclaimer, encoding="utf-8")
    return out_png

# ===================== 엔트리 =====================
import argparse

async def start_trading():
    # 하드코딩된 API 키 사용
    ex = ccxt_async.bithumb({
        "apiKey": BITHUMB_API_KEY,
        "secret": BITHUMB_API_SECRET,
        "enableRateLimit": True,
    })

    bot = Bot(ex)
    LOG.info("설정", **{k:getattr(CFG,k) for k in CFG.__dataclass_fields__.keys() if k not in {"deepseek_key","newsapi_key"}})
    try:
        await bot.run()
    finally:
        await ex.close()

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--symbol", default="BTC/KRW")
    parser.add_argument("--start", default=(datetime.now(KST)-timedelta(days=365)).strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=datetime.now(KST).strftime("%Y-%m-%d"))
    parser.add_argument("--k-list", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--sl-list", default="1.0,1.5,2.0")
    args = parser.parse_args()

    if args.backtest:
        ex = ccxt_async.bithumb({"enableRateLimit": True})
        try:
            out = await backtest_grid(ex, args.symbol, args.start, args.end,
                                      [float(x) for x in args.k_list.split(",")],
                                      [float(x) for x in args.sl_list.split(",")])
            print(f"그리드 리포트 저장: {out}")
        finally:
            await ex.close()
    else:
        await start_trading()

if __name__ == "__main__":
    asyncio.run(main_async())
