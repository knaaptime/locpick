"""Formatting utilities for choice model estimation results.

This module provides functions for formatting coefficient tables, fit
statistics, and side-by-side model comparisons in plain text, HTML, and LaTeX.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def format_coefficient_table(
    result,
    format: str = "text",
    significance_codes: bool = True,
    alpha_levels: Optional[dict] = None,
) -> str:
    """Format a coefficient table from a FitResult.

    Parameters
    ----------
    result : FitResult
        Estimation results to format.
    format : str
        Output format: ``"text"``, ``"html"``, or ``"latex"``.
    significance_codes : bool
        Whether to append significance stars.
    alpha_levels : dict, optional
        Mapping of star symbols to significance levels.
        Default: ``{"***": 0.01, "**": 0.05, "*": 0.1}``.

    Returns
    -------
    str
        Formatted coefficient table.
    """
    if alpha_levels is None:
        alpha_levels = {"***": 0.01, "**": 0.05, "*": 0.1}

    tidy = result.tidy()

    if significance_codes:
        stars = []
        for p in tidy["p_value"]:
            star = ""
            for symbol, alpha in sorted(alpha_levels.items(), key=lambda x: x[1]):
                if p <= alpha:
                    star = symbol
                    break
            stars.append(star)
        tidy["signif"] = stars

    if format == "html":
        return _format_coefficient_html(tidy, significance_codes)
    elif format == "latex":
        return _format_coefficient_latex(tidy, significance_codes)
    else:
        return _format_coefficient_text(tidy, significance_codes)


def format_fit_statistics(
    result,
    format: str = "text",
) -> str:
    """Format model fit statistics from a FitResult.

    Parameters
    ----------
    result : FitResult
        Estimation results to format.
    format : str
        Output format: ``"text"``, ``"html"``, or ``"latex"``.

    Returns
    -------
    str
        Formatted fit statistics.
    """
    stats_df = result.fit_statistics()

    if format == "html":
        rows = []
        for _, row in stats_df.iterrows():
            val = row["value"]
            if isinstance(val, float):
                rows.append(f"<tr><td>{row['statistic']}</td><td>{val:.4f}</td></tr>")
            else:
                rows.append(f"<tr><td>{row['statistic']}</td><td>{val}</td></tr>")
        return "<table>\n" + "\n".join(rows) + "\n</table>"
    elif format == "latex":
        rows = []
        for _, row in stats_df.iterrows():
            val = row["value"]
            if isinstance(val, float):
                rows.append(f"{row['statistic']} & {val:.4f} \\\\")
            else:
                rows.append(f"{row['statistic']} & {val} \\\\")
        return "\\begin{tabular}{ll}\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}"
    else:
        lines = []
        for _, row in stats_df.iterrows():
            val = row["value"]
            if isinstance(val, float):
                lines.append(f"{row['statistic']:<25} {val:>12.4f}")
            else:
                lines.append(f"{row['statistic']:<25} {val:>12}")
        return "\n".join(lines)


def format_side_by_side(
    results: list,
    labels: Optional[list[str]] = None,
    format: str = "text",
) -> str:
    """Format multiple FitResults side by side for comparison.

    Parameters
    ----------
    results : list of FitResult
        Estimation results to compare.
    labels : list of str, optional
        Labels for each model. Default: Model 1, Model 2, ...
    format : str
        Output format: ``"text"``, ``"html"``, or ``"latex"``.

    Returns
    -------
    str
        Side-by-side comparison table.
    """
    if labels is None:
        labels = [f"Model {i + 1}" for i in range(len(results))]

    if format == "html":
        return _format_side_by_side_html(results, labels)
    elif format == "latex":
        return _format_side_by_side_latex(results, labels)
    else:
        return _format_side_by_side_text(results, labels)


# ------------------------------------------------------------------
# Internal formatters
# ------------------------------------------------------------------


def _format_coefficient_text(tidy: pd.DataFrame, significance_codes: bool) -> str:
    lines = []
    header = f"{'Parameter':<20} {'Coef':>10} {'Std.Err':>10} {'t':>8} {'P>|t|':>8}"
    if significance_codes:
        header += "   "
    lines.append(header)
    lines.append("-" * len(header))

    for name, row in tidy.iterrows():
        line = (
            f"{name:<20} {row['coefficient']:>10.4f} "
            f"{row['std_error']:>10.4f} {row['t_value']:>8.3f} "
            f"{row['p_value']:>8.4f}"
        )
        if significance_codes:
            line += f"   {row.get('signif', '')}"
        lines.append(line)

    return "\n".join(lines)


def _format_coefficient_html(tidy: pd.DataFrame, significance_codes: bool) -> str:
    header = "<tr><th>Parameter</th><th>Coef</th><th>Std.Err</th>"
    header += "<th>t</th><th>P>|t|</th>"
    if significance_codes:
        header += "<th></th>"
    header += "</tr>"

    rows = []
    for name, row in tidy.iterrows():
        row_html = (
            f"<tr><td>{name}</td><td>{row['coefficient']:.4f}</td>"
            f"<td>{row['std_error']:.4f}</td>"
            f"<td>{row['t_value']:.3f}</td>"
            f"<td>{row['p_value']:.4f}</td>"
        )
        if significance_codes:
            row_html += f"<td>{row.get('signif', '')}</td>"
        row_html += "</tr>"
        rows.append(row_html)

    return f"<table>\n{header}\n" + "\n".join(rows) + "\n</table>"


def _format_coefficient_latex(tidy: pd.DataFrame, significance_codes: bool) -> str:
    header = "Parameter & Coef & Std.Err & t & P>|t|"
    if significance_codes:
        header += " & "
    header += " \\\\"
    lines = [
        "\\begin{tabular}{lcccc" + ("c" if significance_codes else "") + "}",
        "\\hline",
        header,
        "\\hline",
    ]

    for name, row in tidy.iterrows():
        line = (
            f"{name} & {row['coefficient']:.4f} & {row['std_error']:.4f} "
            f"& {row['t_value']:.3f} & {row['p_value']:.4f}"
        )
        if significance_codes:
            line += f" & {row.get('signif', '')}"
        line += " \\\\"
        lines.append(line)

    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def _format_side_by_side_text(results: list, labels: list[str]) -> str:
    # Collect all parameter names
    all_params = []
    for r in results:
        for p in r.coefficients.index:
            if p not in all_params:
                all_params.append(p)

    len(results)
    col_width = 12
    header = f"{'Parameter':<20}" + "".join(f" {l:>{col_width}}" for l in labels)
    lines = [header, "-" * len(header)]

    for param in all_params:
        row = f"{param:<20}"
        for r in results:
            if param in r.coefficients.index:
                row += f" {r.coefficients[param]:>{col_width}.4f}"
            else:
                row += f" {'—':>{col_width}}"
        lines.append(row)

    # Fit statistics
    lines.append("-" * len(header))
    stats_rows = [
        ("LL", lambda r: f"{r.log_likelihood:.4f}"),
        ("AIC", lambda r: f"{r.aic:.4f}"),
        ("BIC", lambda r: f"{r.bic:.4f}"),
        ("Rho²", lambda r: f"{r.rho_squared:.4f}"),
        ("N", lambda r: str(r.n_observations)),
        ("k", lambda r: str(r.n_parameters)),
    ]
    for stat_name, fmt_fn in stats_rows:
        row = f"{stat_name:<20}"
        for r in results:
            row += f" {fmt_fn(r):>{col_width}}"
        lines.append(row)

    return "\n".join(lines)


def _format_side_by_side_html(results: list, labels: list[str]) -> str:
    all_params = []
    for r in results:
        for p in r.coefficients.index:
            if p not in all_params:
                all_params.append(p)

    header = "<tr><th>Parameter</th>" + "".join(f"<th>{l}</th>" for l in labels) + "</tr>"
    rows = []
    for param in all_params:
        row = f"<tr><td>{param}</td>"
        for r in results:
            if param in r.coefficients.index:
                row += f"<td>{r.coefficients[param]:.4f}</td>"
            else:
                row += "<td>—</td>"
        row += "</tr>"
        rows.append(row)

    return f"<table>\n{header}\n" + "\n".join(rows) + "\n</table>"


def _format_side_by_side_latex(results: list, labels: list[str]) -> str:
    all_params = []
    for r in results:
        for p in r.coefficients.index:
            if p not in all_params:
                all_params.append(p)

    1 + len(results)
    col_spec = "l" + "r" * len(results)
    header = "Parameter & " + " & ".join(labels) + " \\\\"
    lines = [f"\\begin{{tabular}}{{{col_spec}}}", "\\hline", header, "\\hline"]

    for param in all_params:
        row = param
        for r in results:
            if param in r.coefficients.index:
                row += f" & {r.coefficients[param]:.4f}"
            else:
                row += " & —"
        row += " \\\\"
        lines.append(row)

    lines.extend(["\\hline", "\\end{tabular}"])
