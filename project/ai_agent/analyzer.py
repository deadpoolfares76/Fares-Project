"""
AI Agent Module
- Mode LOCAL  : analyse règle-based automatique, sans API (toujours disponible)
- Mode CLAUDE : analyse enrichie via l'API Anthropic (optionnelle)
"""
import os
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _f(val, dec=2, pct=False) -> str:
    try:
        v = float(val)
        if np.isnan(v):
            return "N/A"
        return f"{v:.{dec}f}{'%' if pct else ''}"
    except (TypeError, ValueError):
        return "N/A"


def _cmp(e, c, higher_is_better=True) -> str:
    try:
        ev, cv = float(e), float(c)
        if np.isnan(ev) or np.isnan(cv):
            return "indéterminé"
        if higher_is_better:
            return "✅ ESG" if ev > cv else ("🟰 Équivalent" if abs(ev-cv) < 1e-6 else "📊 Classique")
        else:
            return "✅ ESG" if ev < cv else ("🟰 Équivalent" if abs(ev-cv) < 1e-6 else "📊 Classique")
    except Exception:
        return "indéterminé"


def _sig(pvalue) -> str:
    try:
        p = float(pvalue)
        if p < 0.01:  return "très significatif (p < 1%)"
        if p < 0.05:  return "significatif (p < 5%)"
        if p < 0.10:  return "marginalement significatif (p < 10%)"
        return f"non significatif (p = {p:.3f})"
    except Exception:
        return "N/A"


def _sharpe_label(sh):
    try:
        v = float(sh)
        if np.isnan(v):    return "non calculable"
        if v > 1.5:        return "excellent"
        if v > 1.0:        return "bon"
        if v > 0.5:        return "acceptable"
        if v > 0.0:        return "faible"
        return "négatif"
    except Exception:
        return "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE LOCALE COMPLÈTE (sans API)
# ─────────────────────────────────────────────────────────────────────────────
def generate_local_analysis(esg_summary: dict,
                              classic_summary: dict,
                              esg_name: str,
                              classic_name: str,
                              ff_results: dict = None,
                              period: str = "période sélectionnée") -> str:
    e, c = esg_summary, classic_summary

    def g(d, k): 
        v = d.get(k, np.nan)
        try: return float(v) if v is not None else np.nan
        except: return np.nan

    e_ret, c_ret   = g(e,"Ann. Return (%)"), g(c,"Ann. Return (%)")
    e_vol, c_vol   = g(e,"Ann. Volatility (%)"), g(c,"Ann. Volatility (%)")
    e_sh,  c_sh    = g(e,"Sharpe Ratio"), g(c,"Sharpe Ratio")
    e_so,  c_so    = g(e,"Sortino Ratio"), g(c,"Sortino Ratio")
    e_tr,  c_tr    = g(e,"Treynor Ratio"), g(c,"Treynor Ratio")
    e_beta, c_beta = g(e,"Beta"), g(c,"Beta")
    e_mdd, c_mdd   = g(e,"Max Drawdown (%)"), g(c,"Max Drawdown (%)")
    e_alpha        = g(e,"CAPM Alpha (%)")
    e_skew, c_skew = g(e,"Skewness"), g(c,"Skewness")
    e_kurt, c_kurt = g(e,"Excess Kurtosis"), g(c,"Excess Kurtosis")
    e_tstat        = g(e,"Alpha t-stat")

    var_key  = next((k for k in e if "VaR"  in k), None)
    cvar_key = next((k for k in e if "CVaR" in k), None)
    e_var  = g(e, var_key)  if var_key  else np.nan
    c_var  = g(c, var_key)  if var_key  else np.nan
    e_cvar = g(e, cvar_key) if cvar_key else np.nan
    c_cvar = g(c, cvar_key) if cvar_key else np.nan

    def diff(a, b): return a - b if not (np.isnan(a) or np.isnan(b)) else np.nan
    ret_diff = diff(e_ret, c_ret)
    sh_diff  = diff(e_sh, c_sh)
    mdd_diff = diff(e_mdd, c_mdd)

    # ── Section 1
    ret_comment = ""
    if not np.isnan(ret_diff):
        if abs(ret_diff) < 0.5:
            ret_comment = "> 🟰 Rendements **quasi-identiques** (écart < 0.5%). La contrainte ESG n'impacte pas le rendement brut sur cette période."
        elif ret_diff > 0:
            ret_comment = f"> ✅ L'ESG **surperforme** de **{_f(ret_diff)}%** en rendement annualisé."
        else:
            ret_comment = f"> ⚠️ Le classique **surperforme** de **{_f(abs(ret_diff))}%**. Coût d'opportunité ESG mesurable."

    s1 = f"""## 📊 1. Performance Absolue — {period}

| Indicateur | {esg_name} | {classic_name} | Écart |
|---|---|---|---|
| Rendement annualisé | **{_f(e_ret)}%** | **{_f(c_ret)}%** | {_f(ret_diff)}% |
| Volatilité annualisée | {_f(e_vol)}% | {_f(c_vol)}% | {_f(diff(e_vol,c_vol))}% |
| Beta (CAPM) | {_f(e_beta,3)} | {_f(c_beta,3)} | — |

{ret_comment}

"""

    # ── Section 2
    sh_comment = ""
    if not np.isnan(sh_diff):
        if abs(sh_diff) < 0.10:
            sh_comment = "> 🟰 Ratio de Sharpe **statistiquement équivalent**. La contrainte ESG ne modifie pas l'efficience risque/rendement."
        elif sh_diff > 0:
            sh_comment = f"> ✅ Meilleur Sharpe ESG (+{_f(sh_diff,3)}). L'investisseur ESG est mieux rémunéré par unité de risque."
        else:
            sh_comment = f"> ⚠️ Meilleur Sharpe Classique (+{_f(abs(sh_diff),3)}). Coût d'efficience de la contrainte ESG."

    s2 = f"""## ⚖️ 2. Rendement Ajusté du Risque

| Ratio | {esg_name} | {classic_name} | Meilleur |
|---|---|---|---|
| **Sharpe** | {_f(e_sh,3)} ({_sharpe_label(e_sh)}) | {_f(c_sh,3)} ({_sharpe_label(c_sh)}) | {_cmp(e_sh,c_sh)} |
| **Sortino** | {_f(e_so,3)} | {_f(c_so,3)} | {_cmp(e_so,c_so)} |
| **Treynor** | {_f(e_tr,4)} | {_f(c_tr,4)} | {_cmp(e_tr,c_tr)} |

{sh_comment}

"""

    # ── Section 3
    mdd_comment = ""
    if not np.isnan(mdd_diff):
        if e_mdd > c_mdd:  # less negative = better
            mdd_comment = f"> ✅ **Protection baissière ESG** : drawdown max ESG ({_f(e_mdd)}%) moins sévère que le classique ({_f(c_mdd)}%). Les filtres ESG éliminent les entreprises à risque idiosyncratique élevé."
        else:
            mdd_comment = f"> ⚠️ Drawdown ESG ({_f(e_mdd)}%) plus important. L'exposition sectorielle ESG (ex: surpondération tech) peut amplifier les pertes en crise."

    kurt_comment = ""
    if not np.isnan(e_kurt) and e_kurt > 1.5:
        kurt_comment = f"\n> ⚠️ **Leptokurtisme ESG** (kurtosis excès = {_f(e_kurt,2)}) : queues épaisses — les modèles gaussiens sous-estiment le risque réel de perte extrême."

    s3 = f"""## 📉 3. Risque Baissier et Pertes Extrêmes

| Mesure | {esg_name} | {classic_name} | Protection ESG |
|---|---|---|---|
| **Max Drawdown** | {_f(e_mdd)}% | {_f(c_mdd)}% | {_cmp(e_mdd,c_mdd,False)} |
| **VaR 95%** | {_f(e_var)}% | {_f(c_var)}% | {_cmp(e_var,c_var,False)} |
| **CVaR 95%** | {_f(e_cvar)}% | {_f(c_cvar)}% | {_cmp(e_cvar,c_cvar,False)} |
| **Skewness** | {_f(e_skew,3)} | {_f(c_skew,3)} | — |
| **Kurtosis excès** | {_f(e_kurt,3)} | {_f(c_kurt,3)} | — |

{mdd_comment}{kurt_comment}

"""

    # ── Section 4
    alpha_comment = ""
    if not np.isnan(e_alpha):
        sig_str = " (**non significatif**, |t| < 2)" if (not np.isnan(e_tstat) and abs(e_tstat) < 2) else " (**significatif**, |t| ≥ 2)"
        sign = "positif" if e_alpha > 0 else "négatif"
        alpha_comment = f"> Alpha CAPM ESG **{sign}** ({_f(e_alpha)}%/an){sig_str}."
        if not np.isnan(e_beta):
            if e_beta < 0.9:
                alpha_comment += f" Beta de {_f(e_beta,3)} — **exposition systématique réduite**, typique des portefeuilles ESG (biais qualité, faible levier)."
            elif e_beta > 1.1:
                alpha_comment += f" Beta de {_f(e_beta,3)} — exposition amplifiée au marché, à surveiller en baisse."
            else:
                alpha_comment += f" Beta de {_f(e_beta,3)} — exposition systématique similaire au marché."

    s4 = f"""## 🔬 4. Alpha CAPM et Exposition Systématique

| | {esg_name} | {classic_name} |
|---|---|---|
| **Alpha annualisé** | {_f(e_alpha)}% | {_f(g(c,'CAPM Alpha (%)'))}% |
| **Beta** | {_f(e_beta,3)} | {_f(c_beta,3)} |
| **t-stat alpha** | {_f(e_tstat,3)} | — |

{alpha_comment}

"""

    # ── Section 5 : Facteurs
    s5 = ""
    if ff_results:
        s5 = "## 🧮 5. Modèles Multi-Facteurs\n\n"
        for model_name, res in ff_results.items():
            if not isinstance(res, dict) or "alpha_annual_pct" not in res:
                continue
            al   = float(res.get("alpha_annual_pct", np.nan) or np.nan)
            at   = float(res.get("alpha_tstat",      np.nan) or np.nan)
            ap   = float(res.get("alpha_pvalue",     np.nan) or np.nan)
            r2   = float(res.get("r_squared",        np.nan) or np.nan)
            sig  = res.get("alpha_significant", False)

            s5 += (f"**{model_name}** — Alpha : {_f(al)}%/an | "
                   f"t-stat : {_f(at,3)} | {_sig(ap)} | R² : {_f(r2,3)}\n\n")

            notable = []
            for fname, fd in res.get("factors", {}).items():
                load = float(fd.get("loading", np.nan) or np.nan)
                fsig = fd.get("significant", False)
                if fsig and not np.isnan(load):
                    d = "positif" if load > 0 else "négatif"
                    if fname == "SMB":
                        notable.append(f"**SMB {d}** ({_f(load,3)}) → {'biais small-cap' if load>0 else 'biais large-cap (typique ESG)'}")
                    elif fname == "HML":
                        notable.append(f"**HML {d}** ({_f(load,3)}) → {'biais value' if load>0 else 'biais croissance (excl. fossiles/tabac)'}")
                    elif fname == "RMW":
                        notable.append(f"**RMW {d}** ({_f(load,3)}) → {'biais qualité/profitabilité ✅' if load>0 else 'biais peu profitable'}")
                    elif fname == "CMA":
                        notable.append(f"**CMA {d}** ({_f(load,3)}) → {'investissement conservateur' if load>0 else 'investissement agressif'}")
                    elif fname in ("Mom","MOM"):
                        notable.append(f"**Momentum {d}** ({_f(load,3)}) → {'performance ESG momentum-driven' if load>0 else 'contra-momentum'}")

            if notable:
                s5 += "Expositions factorielles significatives :\n"
                for n in notable:
                    s5 += f"- {n}\n"
                s5 += "\n"

            if sig and not np.isnan(al) and al > 0:
                s5 += "> ✅ **Alpha ESG positif et significatif** après contrôle factoriel.\n\n"
            elif not sig:
                s5 += "> ℹ️ Alpha non significatif — la performance s'explique par les expositions factorielles.\n\n"

    # ── Section 6 : Verdict
    wins, total = 0, 0
    for ev, cv, hib in [(e_sh,c_sh,True),(e_so,c_so,True),(e_tr,c_tr,True),
                         (e_ret,c_ret,True),(e_mdd,c_mdd,False),(e_var,c_var,False)]:
        try:
            if not (np.isnan(float(ev)) or np.isnan(float(cv))):
                total += 1
                if (hib and float(ev) > float(cv)) or (not hib and float(ev) > float(cv)):
                    wins += 1
        except Exception:
            pass

    if total > 0:
        pct = wins / total * 100
        if pct >= 60:
            verdict = "✅ **Favorable à l'ESG**"
            v_text  = "Le portefeuille ESG domine sur la majorité des métriques. La contrainte ESG n'a pas détérioré le profil risque/rendement."
        elif pct >= 40:
            verdict = "⚖️ **Résultats mixtes**"
            v_text  = "Résultats partagés. La contrainte ESG crée des trade-offs selon la dimension analysée."
        else:
            verdict = "⚠️ **Favorable au Classique**"
            v_text  = "Le portefeuille classique domine sur la période. Coût d'opportunité ESG mesurable."
    else:
        verdict = "⚠️ Données insuffisantes"
        v_text  = ""

    s6 = f"""## 🏁 6. Verdict & Recommandations Institutionnelles

**{verdict}**

{v_text}

### Points de vigilance
1. **Horizon** : une seule fenêtre temporelle ne suffit pas — analyser sur plusieurs sous-périodes.
2. **Biais sectoriel** : les ETFs ESG surpondèrent souvent la tech et sous-pondèrent l'énergie.
3. **Définition ESG** : les résultats varient selon le fournisseur (MSCI vs Sustainalytics vs ISS).
4. **Significativité** : sans 10+ ans de données, les différences de Sharpe restent rarement significatives.
5. **Coûts** : les portefeuilles ESG ont un TER et un turnover légèrement supérieurs.

---
*🔢 Analyse générée automatiquement par le moteur local — sans clé API. Pour une analyse Claude AI, entrez votre clé Anthropic dans la sidebar.*
"""

    return s1 + s2 + s3 + s4 + s5 + s6


# ─────────────────────────────────────────────────────────────────────────────
# TEST DE CONNEXION API
# ─────────────────────────────────────────────────────────────────────────────
def test_api_connection(api_key: str) -> tuple[bool, str]:
    """
    Test the Anthropic API key with a minimal request.
    Returns (success, message).
    """
    if not api_key or not api_key.strip():
        return False, "Aucune clé API fournie."

    key = api_key.strip()
    if not key.startswith("sk-ant-"):
        return False, f"Format de clé invalide (doit commencer par 'sk-ant-'). Reçu : '{key[:10]}...'"

    try:
        import anthropic
    except ImportError:
        return False, "Package 'anthropic' non installé. Lancez : pip install anthropic"

    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        return True, f"✅ Connexion réussie (modèle : {msg.model})"
    except anthropic.AuthenticationError:
        return False, "❌ Clé API invalide ou révoquée. Vérifiez sur console.anthropic.com"
    except anthropic.PermissionDeniedError:
        return False, "❌ Accès refusé. La clé n'a pas les permissions nécessaires."
    except anthropic.RateLimitError:
        return False, "⚠️ Limite de débit atteinte. Réessayez dans quelques secondes."
    except Exception as e:
        return False, f"❌ Erreur inattendue : {type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE VIA CLAUDE API
# ─────────────────────────────────────────────────────────────────────────────
def _build_api_prompt(esg_summary, classic_summary, esg_name, classic_name,
                       ff_results=None, period="") -> str:
    def fmt(s):
        return "\n".join(
            f"  - {k}: {_f(v)}"
            for k, v in s.items() if k != "Name"
        )

    prompt = (
        f"Tu es un analyste quantitatif senior spécialisé en investissement ESG. "
        f"Fournis une analyse rigoureuse en français comparant ces deux portefeuilles sur {period}.\n\n"
        f"## {esg_name} (ESG)\n{fmt(esg_summary)}\n\n"
        f"## {classic_name} (Classique)\n{fmt(classic_summary)}\n"
    )

    if ff_results:
        prompt += "\n## Régressions Multi-Facteurs\n"
        for mn, res in ff_results.items():
            if isinstance(res, dict) and "alpha_annual_pct" in res:
                sig = "✓ Significatif" if res.get("alpha_significant") else "✗ Non significatif"
                prompt += (
                    f"\n### {mn}\n"
                    f"  Alpha: {_f(res['alpha_annual_pct'])}%/an [{sig}], "
                    f"t={_f(res.get('alpha_tstat'),3)}, R²={_f(res.get('r_squared'),3)}\n"
                )
                for fn, fd in res.get("factors", {}).items():
                    prompt += f"  {fn}: β={_f(fd['loading'],3)} (t={_f(fd['t_stat'],3)})\n"

    prompt += (
        "\n## Analyse requise (markdown, français)\n"
        "1. Performance ajustée du risque — différence économiquement significative ?\n"
        "2. Prime/pénalité ESG — les facteurs expliquent-ils la performance ?\n"
        "3. Protection baissière — VaR, CVaR, drawdown\n"
        "4. Expositions factorielles — quels facteurs expliquent l'alpha ESG ?\n"
        "5. Significativité statistique — résultats robustes ?\n"
        "6. Recommandation institutionnelle — allocation ESG justifiée ?\n"
        "7. Mises en garde — limites et biais\n\n"
        "Sois précis, cite les chiffres, ton professionnel et équilibré."
    )
    return prompt


def run_ai_analysis(esg_summary: dict,
                     classic_summary: dict,
                     esg_name: str,
                     classic_name: str,
                     ff_results: dict = None,
                     period: str = "période sélectionnée",
                     api_key: str = None,
                     model: str = "claude-sonnet-4-6") -> tuple[str, str]:
    """
    Run analysis.
    Returns (text, source) where source ∈ {"local", "claude_api", "local_fallback"}
    """
    key = (api_key or os.environ.get("ANTHROPIC_API_KEY", "")).strip()

    # No key → local analysis
    if not key:
        return generate_local_analysis(
            esg_summary, classic_summary, esg_name, classic_name, ff_results, period
        ), "local"

    # Has key → try Claude API
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        prompt = _build_api_prompt(
            esg_summary, classic_summary, esg_name, classic_name, ff_results, period
        )
        msg = client.messages.create(
            model=model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text, "claude_api"

    except ImportError:
        local = generate_local_analysis(
            esg_summary, classic_summary, esg_name, classic_name, ff_results, period
        )
        return local + "\n\n---\n⚠️ *Package `anthropic` non installé : `pip install anthropic`*", "local_fallback"

    except Exception as e:
        error_type = type(e).__name__
        error_msg  = str(e)

        # Provide specific guidance per error type
        if "AuthenticationError" in error_type or "401" in error_msg:
            hint = "❌ **Clé API invalide ou révoquée.** Vérifiez sur [console.anthropic.com](https://console.anthropic.com) et créez une nouvelle clé."
        elif "PermissionDenied" in error_type or "403" in error_msg:
            hint = "❌ **Accès refusé.** Votre clé n'a pas les permissions pour ce modèle."
        elif "RateLimit" in error_type or "429" in error_msg:
            hint = "⚠️ **Limite de débit.** Trop de requêtes — attendez quelques secondes et réessayez."
        elif "InsufficientCredit" in error_type or "credit" in error_msg.lower():
            hint = "💳 **Crédit insuffisant.** Rechargez votre compte sur [console.anthropic.com](https://console.anthropic.com/billing)."
        elif "ConnectionError" in error_type or "timeout" in error_msg.lower():
            hint = "🌐 **Erreur réseau.** Vérifiez votre connexion internet."
        else:
            hint = f"Erreur `{error_type}` : {error_msg}"

        local = generate_local_analysis(
            esg_summary, classic_summary, esg_name, classic_name, ff_results, period
        )
        return (
            f"> ⚠️ **API Claude indisponible** — analyse locale affichée.\n> {hint}\n\n---\n\n" + local
        ), "local_fallback"


# ─────────────────────────────────────────────────────────────────────────────
# COMMENTAIRE RÉGIMES
# ─────────────────────────────────────────────────────────────────────────────
def generate_regime_commentary(regime_esg: dict,
                                 regime_classic: dict,
                                 api_key: str = None) -> tuple[str, str]:
    """Returns (commentary, source)."""

    def local_regime(esg_r, cls_r):
        lines = ["## 🌊 Analyse par Régime de Volatilité\n"]
        for regime in ["High Volatility", "Low Volatility"]:
            e = esg_r.get(regime, {})
            c = cls_r.get(regime, {})
            if not e or not c:
                continue
            label = "🌩️ Haute volatilité" if "High" in regime else "📈 Basse volatilité"
            lines += [
                f"### {label} ({e.get('n_days',0)} jours)\n",
                f"| | ESG | Classique |", f"|---|---|---|",
                f"| Rendement annualisé | {_f(e.get('ann_return_pct'))}% | {_f(c.get('ann_return_pct'))}% |",
                f"| Sharpe | {_f(e.get('sharpe'),3)} | {_f(c.get('sharpe'),3)} |",
                f"| Beta | {_f(e.get('beta'),3)} | {_f(c.get('beta'),3)} |",
                f"| Max Drawdown | {_f(e.get('max_dd_pct'))}% | {_f(c.get('max_dd_pct'))}% |\n",
            ]
            try:
                e_sh = float(e.get("sharpe", np.nan) or np.nan)
                c_sh = float(c.get("sharpe", np.nan) or np.nan)
                e_b  = float(e.get("beta",   np.nan) or np.nan)
                c_b  = float(c.get("beta",   np.nan) or np.nan)
                if "High" in regime and not (np.isnan(e_sh) or np.isnan(c_sh)):
                    if e_sh > c_sh:
                        lines.append("> ✅ En période de stress, l'ESG affiche un meilleur Sharpe — cohérent avec la thèse de protection baissière.\n")
                    else:
                        lines.append("> ⚠️ En stress, le classique résiste mieux — diversification sectorielle plus large.\n")
                if not (np.isnan(e_b) or np.isnan(c_b)) and e_b < c_b and "High" in regime:
                    lines.append(f"> 📉 Compression de beta ESG ({_f(e_b,3)}) vs Classique ({_f(c_b,3)}) en régime de stress.\n")
            except Exception:
                pass
            lines.append("")
        return "\n".join(lines)

    key = (api_key or os.environ.get("ANTHROPIC_API_KEY", "")).strip()

    if not key:
        return local_regime(regime_esg, regime_classic), "local"

    try:
        import anthropic
        def fmt_r(d):
            lines = []
            for reg, m in d.items():
                lines.append(f"\n{reg} ({m.get('n_days',0)} jours):")
                for k, v in m.items():
                    if k != "n_days":
                        lines.append(f"  {k}: {_f(v)}")
            return "\n".join(lines)

        prompt = (
            f"Analyse ces résultats de régimes de volatilité en 3-4 paragraphes (français, markdown).\n\n"
            f"ESG:\n{fmt_r(regime_esg)}\n\nClassique:\n{fmt_r(regime_classic)}\n\n"
            f"Couvre: (1) comportement ESG en stress, (2) compression du beta, "
            f"(3) protection baissière, (4) implications pratiques."
        )
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text, "claude_api"
    except Exception:
        return local_regime(regime_esg, regime_classic), "local"
