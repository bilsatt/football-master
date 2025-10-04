import math

def _poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def _poisson_matrix(lh, la, max_goals=6):
    pm = []
    for i in range(max_goals+1):
        row = []
        for j in range(max_goals+1):
            row.append(_poisson_pmf(i, lh) * _poisson_pmf(j, la))
        pm.append(row)
    return pm

def _result_probs(lh, la, max_goals=6):
    m = _poisson_matrix(lh, la, max_goals)
    pH = sum(m[i][j] for i in range(max_goals+1) for j in range(max_goals+1) if i>j)
    pD = sum(m[i][i] for i in range(max_goals+1))
    pA = 1.0 - pH - pD
    return {'H': pH, 'D': pD, 'A': pA}

def implied_from_odds(odds_map):
    # odds_map like {'Home': 1.90, 'Draw': 3.30, 'Away': 4.20}
    if not odds_map:
        return None
    inv = {}
    total = 0.0
    for k,v in odds_map.items():
        try:
            iv = 1.0/float(v)
        except Exception:
            iv = 0.0
        inv[k] = iv; total += iv
    if total <= 0:
        return None
    # normalize to sum=1 (remove overround roughly)
    norm = {k: inv[k]/total for k in inv}
    def mapkey(k):
        return {'Home':'H', 'Draw':'D', 'Away':'A'}.get(k, k)
    return {mapkey(k): v for k,v in norm.items() if mapkey(k) in ('H','D','A')}

def predict_match(api_get, home_id, away_id, league_id, season,
                  last5_home_for=1.2, last5_home_against=1.0,
                  last5_away_for=1.1, last5_away_against=1.2,
                  odds_map=None, alpha=0.8, max_goals=6):
    # Returns dict with lambda_home, lambda_away, probs_dc (model only) and probs_final (blended).
    # alpha blends market and model: final = alpha*market + (1-alpha)*model
    lam_home = 0.55*max(0.05, last5_home_for) + 0.45*max(0.05, last5_away_against)
    lam_away = 0.55*max(0.05, last5_away_for) + 0.45*max(0.05, last5_home_against)
    lam_home *= 1.07  # light home advantage
    probs_dc = _result_probs(lam_home, lam_away, max_goals=max_goals)
    market_probs = implied_from_odds(odds_map) if odds_map else None
    if market_probs:
        pH = (1-alpha)*probs_dc['H'] + alpha*market_probs.get('H', probs_dc['H'])
        pD = (1-alpha)*probs_dc['D'] + alpha*market_probs.get('D', probs_dc['D'])
        pA = (1-alpha)*probs_dc['A'] + alpha*market_probs.get('A', probs_dc['A'])
        s = max(pH+pD+pA, 1e-9)
        probs_final = {'H': pH/s, 'D': pD/s, 'A': pA/s}
    else:
        probs_final = probs_dc
    return {
        'lambda_home': lam_home,
        'lambda_away': lam_away,
        'probs_dc': probs_dc,
        'probs_final': probs_final
    }