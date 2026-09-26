"""High-fidelity synthetic L2 event stream (offline, deterministic).

The generator is built around three latent variables that reproduce the
microstructure economics this project studies:

  * ``fair``  -- the latent efficient price. It is driven by the *permanent
    impact* of signed order flow plus a small regime drift and noise.
  * ``z``     -- latent aggressor pressure, a persistent AR(1) process with
    occasional sign-flipping shocks. Market-order signs follow ``z``, so buys
    and sells cluster the way they do on real venues (trade autocorrelation).
  * ``qmid``  -- the mid around which market makers actually quote. It tracks
    ``fair`` with a lag (partial adjustment). The lag is the single feature
    that creates every effect we measure:
      - order flow now -> fair moves -> quoted mid glues after, so flow/OFI
        predict the mid over the next few events (apparent alpha);
      - a resting bid is hit by clustered sell pressure, exactly when fair is
        being pushed down and qmid is about to glue down (adverse selection);
      - a delayed participant acts on a stale qmid (latency cost).

The recorded book is always a valid, tight grid around qmid. Levels persist
when the best moves (our resting orders are not auto-cancelled by a one-tick
shift); only trades remove top-of-book size and only levels that leave the
grid are cancelled.

Output schema matches data.binance_client exactly, so live and synthetic
data are interchangeable downstream.
"""
from __future__ import annotations

import argparse
import json
import math
import random

import config


def _round_tick(x: float, tick: float) -> float:
    return round(round(x / tick) * tick, 10)


class SyntheticLOB:
    def __init__(self, n_events: int = config.SYNTH_EVENTS,
                 seed: int = config.SYNTH_SEED,
                 tick: float = config.TICK_SIZE,
                 start_price: float = 83_900.0):
        self.n = n_events
        self.rng = random.Random(seed)
        self.tick = tick
        self.fair = start_price
        self.qmid = start_price
        self.z = 0.0
        self.bb = _round_tick(start_price - tick / 2, tick)
        self.ba = self.bb + tick
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ts = 1_700_000_000_000
        self.update_id = 1
        self.drift_regime = 0.0
        self._populate()

    # -- distributions -----------------------------------------------------
    def _qty(self, mean: float, sigma: float = 0.6) -> float:
        x = math.exp(self.rng.gauss(math.log(mean), sigma))
        return round(max(0.0001, x), 5)

    def _level_qty(self, depth_i: int) -> float:
        # depth grows slightly thicker away from the touch
        return self._qty(1.3 + 0.18 * depth_i, sigma=0.5)

    def _populate(self) -> None:
        m = int(round(self.qmid / self.tick))
        self.bb = round(m * self.tick, 10)
        self.ba = self.bb + self.tick
        for i in range(40):
            self.bids[round((m - i) * self.tick, 10)] = self._level_qty(i)
            self.asks[round((m + 1 + i) * self.tick, 10)] = self._level_qty(i)

    # -- helpers -----------------------------------------------------------
    def _set_level(self, side: str, price: float, qty: float,
                   changed: dict[tuple[str, float], float]) -> None:
        book = self.bids if side == "bid" else self.asks
        if qty <= 1e-9:
            book.pop(price, None)
            changed[(side, price)] = 0.0
        else:
            book[price] = round(qty, 5)
            changed[(side, price)] = round(qty, 5)

    def _advance_time(self) -> None:
        # sub-100ms interarrival distribution so latency sweeps are meaningful
        self.ts += max(1, int(self.rng.expovariate(1 / 90.0)))

    # -- latent state ------------------------------------------------------
    def _update_pressure(self) -> None:
        self.z = 0.92 * self.z + self.rng.gauss(0, 0.45)
        if self.rng.random() < 0.015:          # regime shock, often flips sign
            self.z += self.rng.gauss(0, 3.0)
        self.z = max(-6.0, min(6.0, self.z))

    def _evolve_fair_baseline(self) -> None:
        if self.rng.random() < 0.01:
            self.drift_regime = self.rng.gauss(0, 0.4) * self.tick
        self.fair += self.drift_regime + self.rng.gauss(0, 0.12) * self.tick

    # -- market order ------------------------------------------------------
    def _market_order(self, changed: dict) -> list[dict]:
        p_buy = 1 / (1 + math.exp(-1.6 * self.z))
        is_buy = self.rng.random() < p_buy
        side = "ask" if is_buy else "bid"
        book = self.asks if is_buy else self.bids
        big = self.rng.random() < 0.06
        size = self._qty(2.6 if big else 0.42)
        trades = []
        remaining = size
        total = 0.0
        guard = 0
        while remaining > 1e-9 and guard < 12:
            guard += 1
            prices = sorted(book) if is_buy else sorted(book, reverse=True)
            if not prices:
                break
            px = prices[0]
            avail = book[px]
            take = min(avail, remaining)
            trades.append({
                "t": self.update_id, "p": f"{px:.2f}", "q": f"{take:.5f}",
                "T": self.ts,
                "m": (not is_buy),  # buyer is maker only when aggressor sells
            })
            self._set_level(side, px, avail - take, changed)
            remaining -= take
            total += take
        # permanent impact: signed traded size moves the efficient price.
        # 0.9 tick of fair per base unit traded (coefficient in tick units).
        self.fair += (1.0 if is_buy else -1.0) * 0.9 * self.tick * total
        return trades

    # -- passive microstructure -------------------------------------------
    def _limit_add(self, changed: dict) -> None:
        is_bid = self.rng.random() < 0.5
        side = "bid" if is_bid else "ask"
        best = self.bb if is_bid else self.ba
        offset = self.rng.choices(range(0, 8),
                                  weights=[40, 22, 13, 8, 6, 5, 3, 3])[0]
        px = best - offset * self.tick if is_bid else best + offset * self.tick
        book = self.bids if is_bid else self.asks
        self._set_level(side, px, book.get(px, 0.0) + self._qty(0.5), changed)

    def _cancel(self, changed: dict) -> None:
        is_bid = self.rng.random() < 0.5
        side = "bid" if is_bid else "ask"
        book = self.bids if is_bid else self.asks
        best = self.bb if is_bid else self.ba
        prices = sorted(book, reverse=is_bid)
        weights = [1.0 / (1 + abs(p - best) / self.tick) for p in prices]
        px = self.rng.choices(prices, weights=weights)[0]
        if abs(px - best) < 1e-12:
            frac = self.rng.uniform(0.05, 0.35)     # partial cancel at best
        else:
            frac = self.rng.uniform(0.2, 0.9)
        self._set_level(side, px, book[px] * (1 - frac), changed)

    # -- quoted mid + book resync -----------------------------------------
    def _glue(self) -> None:
        # market makers partially close the gap between quotes and fair
        self.qmid += 0.30 * (self.fair - self.qmid) \
            + self.rng.gauss(0, 0.02) * self.tick

    def _resync(self, changed: dict, n: int = 40) -> None:
        """Maintain a valid grid around qmid with the minimum of churn.

        Existing levels are kept (never silently shrunk); missing grid levels
        are added; levels that leave the grid are cancelled. Comparisons use
        integer tick indices.
        """
        m = int(round(self.qmid / self.tick))
        widen = 1 if (abs(self.z) > 1.7 and self.rng.random() < 0.5) else 0
        spread_t = 1 + widen
        bb_idx = m - spread_t // 2
        ba_idx = bb_idx + spread_t
        want_bid = {bb_idx - i for i in range(n)}
        want_ask = {ba_idx + i for i in range(n)}
        for p in list(self.bids):
            if int(round(p / self.tick)) not in want_bid:
                self._set_level("bid", p, 0.0, changed)
        for p in list(self.asks):
            if int(round(p / self.tick)) not in want_ask:
                self._set_level("ask", p, 0.0, changed)
        for idx in want_bid:
            p = round(idx * self.tick, 10)
            if p not in self.bids:
                self._set_level("bid", p, self._level_qty(abs(bb_idx - idx)),
                                changed)
        for idx in want_ask:
            p = round(idx * self.tick, 10)
            if p not in self.asks:
                self._set_level("ask", p, self._level_qty(abs(idx - ba_idx)),
                                changed)
        self.bb = round(bb_idx * self.tick, 10)
        self.ba = round(ba_idx * self.tick, 10)

    # -- main loop ---------------------------------------------------------
    def lines(self):
        snap = {
            "lastUpdateId": 0,
            "bids": [[f"{p:.2f}", f"{q:.5f}"] for p, q in
                     sorted(self.bids.items(), reverse=True)],
            "asks": [[f"{p:.2f}", f"{q:.5f}"] for p, q in
                     sorted(self.asks.items())],
        }
        yield json.dumps({"type": "snapshot", "ts": self.ts, "data": snap})

        for _ in range(self.n):
            self._advance_time()
            self._update_pressure()
            self._evolve_fair_baseline()

            changed: dict[tuple[str, float], float] = {}
            trades: list[dict] = []
            r = self.rng.random()
            if r < 0.30:
                trades = self._market_order(changed)
            elif r < 0.68:
                self._limit_add(changed)
            else:
                self._cancel(changed)

            self._glue()
            self._resync(changed)

            for tr in trades:
                tr_full = {"e": "trade", "E": tr["T"], "s": "BTCUSDT",
                           "t": tr["t"], "p": tr["p"], "q": tr["q"],
                           "T": tr["T"], "m": tr["m"]}
                yield json.dumps({"type": "trade", "data": tr_full})

            if changed:
                U = self.update_id
                self.update_id += 1
                bids = [[f"{p:.2f}", f"{q:.5f}"]
                        for (s, p), q in changed.items() if s == "bid"]
                asks = [[f"{p:.2f}", f"{q:.5f}"]
                        for (s, p), q in changed.items() if s == "ask"]
                d = {"e": "depthUpdate", "E": self.ts, "s": "BTCUSDT",
                     "U": U, "u": self.update_id, "b": bids, "a": asks}
                yield json.dumps({"type": "depth", "data": d})


def generate(out_path: str, n_events: int = config.SYNTH_EVENTS,
             seed: int = config.SYNTH_SEED) -> str:
    gen = SyntheticLOB(n_events=n_events, seed=seed)
    with open(out_path, "w") as f:
        for line in gen.lines():
            f.write(line + "\n")
    print(f"Generated {n_events} synthetic events -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=config.SYNTH_EVENTS)
    ap.add_argument("--seed", type=int, default=config.SYNTH_SEED)
    ap.add_argument("--out", default=str(config.RAW_DIR / "btcusdt_synth.jsonl"))
    args = ap.parse_args()
    generate(args.out, args.events, args.seed)


if __name__ == "__main__":
    main()
