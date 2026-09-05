"""Plotting helpers for the project slides and notebooks.

Each function returns the matplotlib Figure (and usually also shows it via ).
Plotting is done with existing experiment data; no new algorithm implementation is needed.

If `plots.SAVE_DIR` is set (a directory path), each function also saves the figure
to that directory under a fixed filename that matches the includegraphics commands
in the LaTeX slides. Set it once after import, e.g.:

    import plots
    plots.SAVE_DIR = 'latex_slides_files/figures'
"""

import os
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors


SAVE_DIR = "latex_slides_files/figures"  # set this to a directory path to enable auto-save (e.g. 'latex_slides_files/figures')


def _save(fig, name):
    if SAVE_DIR is None:
        return
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig.savefig(os.path.join(SAVE_DIR, name), dpi=150, bbox_inches='tight')


def _all_phase_change_times(environment):
    """Union of all per-campaign phase change times."""
    times = []
    for campaign in environment.campaigns:
        times.extend(np.asarray(campaign.phase_change_times).tolist())
    return sorted(set(times))


def _budget_exhausted_t_from_avg_payments(experiment):
    """First time at which the *average* cumulative spend has stopped growing.

    The intuition: when the bidder has effectively exhausted its budget across
    trials, the per-trial mean cumulative spend becomes flat. We detect that as
    the first index at which the average cumulative spend equals its terminal
    value (up to numerical tolerance).
    """
    results = experiment.results
    starting_budget = experiment.starting_budget
    if not (np.isfinite(starting_budget) and starting_budget > 0):
        return None
    avg_cum = results["avg_payments"].sum(axis=1)
    final = avg_cum[-1]
    if final <= 0:
        return None
    idxs = np.where(avg_cum >= final - 1e-9)[0]
    return int(idxs[0]) if len(idxs) > 0 else None


# ============================================================
# Generic experiment plot (replaces Experiment.plot_experiment_result)
# ============================================================

def plot_experiment_result(experiment, save_name):
    """Plot the standard experiment results: cumulative payments, regret, pulls.

    When more than one campaign is present, the total cumulative spend is shown
    alongside the per-campaign curves. A second pulls plot excludes the bid 0.0
    (which usually dominates and makes the other pulls indistinguishable). A
    vertical line marks the first time at which the *average* cumulative spend
    settles at its terminal value -- a proxy for when the bidder stops bidding
    because it has exhausted its budget. The same vertical line is also drawn on
    the regret plot.
    """
    results = experiment.results
    n_users = experiment.n_users
    bids_space = experiment.bids_space
    n_campaigns = experiment.n_campaigns
    starting_budget = experiment.starting_budget
    bidder_name = experiment.bidder_class.__name__
    bidder = experiment.bidders[-1]

    budget_exhausted_t = _budget_exhausted_t_from_avg_payments(experiment)
    has_total = n_campaigns > 1

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 0. Cumulative payments (total + per-campaign) with a budget reference line
    if has_total:
        avg_total_cumulative_payments = results["avg_total_cumulative_payments"]
        std_total_cumulative_payments = results["std_total_cumulative_payments"]
        axes[0, 0].plot(np.arange(n_users), avg_total_cumulative_payments, label='Total cumulative spend', color='black')
        axes[0, 0].fill_between(
            np.arange(n_users),
            avg_total_cumulative_payments - std_total_cumulative_payments,
            avg_total_cumulative_payments + std_total_cumulative_payments,
            alpha=0.2, color='black'
        )
    for i in range(n_campaigns):
        axes[0, 0].plot(np.arange(n_users), results["avg_payments"][:, i], label=f'Campaign {i}')
        axes[0, 0].fill_between(
            np.arange(n_users),
            results["avg_payments"][:, i] - results["std_payments"][:, i],
            results["avg_payments"][:, i] + results["std_payments"][:, i],
            alpha=0.2
        )
    axes[0, 0].axhline(starting_budget, color='red', linestyle='--', label='Budget')
    if budget_exhausted_t is not None and budget_exhausted_t < n_users - 1:
        axes[0, 0].axvline(budget_exhausted_t, color='green', linestyle=':', label='Budget exhausted')
    axes[0, 0].set_xlabel('$t$')
    axes[0, 0].set_ylabel('$\\sum c_t$')
    axes[0, 0].legend()
    axes[0, 0].set_title(f'Cumulative Payments of {bidder_name}')

    # 1. Cumulative regret (total across all campaigns), with the same budget-exhausted line
    axes[0, 1].plot(np.arange(n_users), results["avg_regrets"], label='Cumulative regret')
    axes[0, 1].fill_between(
        np.arange(n_users),
        results["avg_regrets"] - results["std_regrets"],
        results["avg_regrets"] + results["std_regrets"],
        alpha=0.2
    )
    if budget_exhausted_t is not None and budget_exhausted_t < n_users - 1:
        axes[0, 1].axvline(budget_exhausted_t, color='green', linestyle=':', label='Budget exhausted')
    axes[0, 1].set_xlabel('$t$')
    axes[0, 1].set_ylabel('$\\sum R_t$')
    axes[0, 1].legend()
    axes[0, 1].set_title(f'Cumulative Regret of {bidder_name}')

    # 2. Chosen bids (by campaign, NaN outside feasibility)
    avg_pulls_full = np.full((n_campaigns, len(bids_space)), np.nan)
    std_pulls_full = np.full((n_campaigns, len(bids_space)), np.nan)
    per_campaign_masks = bidder.feasible  # shape (N_CAMPAIGNS, K)
    for i in range(n_campaigns):
        avg_pulls_full[i, per_campaign_masks[i]] = results["avg_pulls"][i, per_campaign_masks[i]]
        std_pulls_full[i, per_campaign_masks[i]] = results["std_pulls"][i, per_campaign_masks[i]]

    for i in range(n_campaigns):
        axes[1, 0].plot(bids_space, avg_pulls_full[i, :], label=f'Campaign {i}')
        axes[1, 0].fill_between(
            bids_space,
            avg_pulls_full[i, :] - std_pulls_full[i, :],
            avg_pulls_full[i, :] + std_pulls_full[i, :],
            alpha=0.2
        )
    axes[1, 0].set_xlabel('$b$')
    axes[1, 0].set_ylabel('$n(b)$')
    axes[1, 0].legend()
    axes[1, 0].set_title(f'Chosen Bids of {bidder_name} by Campaign')

    # 3. Same pulls plot but excluding the bid 0.0 (often dominates and hides structure)
    for i in range(n_campaigns):
        axes[1, 1].plot(bids_space[1:], avg_pulls_full[i, 1:], label=f'Campaign {i}')
        axes[1, 1].fill_between(
            bids_space[1:],
            avg_pulls_full[i, 1:] - std_pulls_full[i, 1:],
            avg_pulls_full[i, 1:] + std_pulls_full[i, 1:],
            alpha=0.2
        )
    axes[1, 1].set_xlabel('$b$')
    axes[1, 1].set_ylabel('$n(b)$')
    axes[1, 1].legend()
    axes[1, 1].set_title(f'Chosen Bids of {bidder_name} by Campaign (no bid 0.0)')

    plt.tight_layout()
    _save(fig, save_name)

    #return fig


# ============================================================
# Generic per-requirement plots
# ============================================================

def plot_win_probabilities(environment, bids_space, save_name):
    """Plot the win probabilities of each campaign over the bid space.

    Each campaign's `get_win_probabilities` may return a per-phase matrix of shape
    (n_phases, K) -- we overlay all phases in the same axes and label them by the
    phase end time (in rounds) so the figure is readable across all requirement
    environments (stationary, highly non-stationary, slightly non-stationary).
    One figure with one subplot per campaign.
    """
    n_campaigns = environment.N_CAMPAIGNS
    n_cols = min(3, n_campaigns)
    n_rows = (n_campaigns + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for i, campaign in enumerate(environment.campaigns):
        r, c = i // n_cols, i % n_cols
        ax = axes[r][c]
        win_probs = campaign.get_win_probabilities(bids_space)      # shape (n_phases, K) or (1, K)
        phase_change_times = [0] + list(np.asarray(campaign.phase_change_times).tolist()) + [np.inf]
        for phase_idx in range(win_probs.shape[0]):
            t_start = phase_change_times[phase_idx]
            t_end = phase_change_times[phase_idx + 1]
            label = f'phase {phase_idx} ({t_start} ≤ t ≤ {t_end})' if np.isfinite(t_end) else f'phase {phase_idx} ({t_start} ≤ t ≤ end)'
            ax.plot(bids_space, win_probs[phase_idx], label=label)
        ax.set_xlabel('$b$')
        ax.set_ylabel('$p_{\\mathrm{win}}(b)$')
        ax.set_title(f'Win probabilities (campaign {i+1})')
        if len(phase_change_times) > 2:
            ax.legend()
    for j in range(n_campaigns, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r][c].axis('off')
    plt.tight_layout()
    _save(fig, save_name)

    #return fig


def plot_per_round_payment_vs_rho(experiment, save_name, bidder_name=None):
    """Per-round payment c_t compared to the per-round budget rho.

    The total per-round payment c_t = sum over campaigns is averaged across trials
    (mean +/- std band). A red horizontal line marks the per-round budget rho.
    Above the line = overspending within a round; below = underspending.
    """
    results = experiment.results
    n_users = experiment.n_users
    rho = experiment.starting_budget / n_users

    # Per-round payments: results["my_payments"] is a list of (n_users, n_campaigns) arrays, one per trial
    per_round_payments_by_trial = np.array(experiment.results["my_payments"]).sum(axis=-1)      # shape (n_trials, n_users)
    avg_per_round = per_round_payments_by_trial.mean(axis=0)
    std_per_round = per_round_payments_by_trial.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(n_users), avg_per_round, label=r'$c_t$ (avg)')
    ax.fill_between(
        np.arange(n_users),
        avg_per_round - std_per_round,
        avg_per_round + std_per_round,
        alpha=0.2, label=r'$c_t$ (std)'
    )
    ax.axhline(rho, color='red', linestyle='--', label=r'$\rho$')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$c_t$')
    title_suffix = f' of {bidder_name}' if bidder_name is not None else ''
    ax.set_title(f'Per-round payment vs $\\rho${title_suffix}')
    ax.legend()
    plt.tight_layout()
    _save(fig, save_name)

    #return fig


def plot_conflict_graph_between_campaigns(environment, save_name):
    """Plot of the conflict graph between campaigns.

    Renders the conflicts as a node-link diagram if networkx is available,
    otherwise falls back to the matrix representation. Connected components are
    placed on concentric circles so the layout is not collapsed into a corner.
    """
    try:
        import networkx as nx
    except ImportError:
        return plot_conflict_graph_between_campaigns_matrix_version(environment)

    cg = environment.conflicts_graph
    G = nx.Graph()
    G.add_nodes_from(range(cg.n_campaigns))
    for (a, b) in cg.graph:
        G.add_edge(a, b)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Place each connected component on its own ring to keep the layout centered
    # and avoid the spring layout corner artifacts.
    components = list(nx.connected_components(G))
    pos = {}
    for comp_idx, comp in enumerate(components):
        nodes = sorted(comp)
        radius = 0.4 + 0.25 * comp_idx  # each component gets a larger ring
        center = (0.0, 0.0) if comp_idx == 0 else (0.6 * (1 if comp_idx % 2 else -1), 0.6 * (1 if comp_idx % 3 == 0 else -1))
        for k, node in enumerate(nodes):
            theta = 2 * np.pi * k / max(1, len(nodes))
            pos[node] = (center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta))

    # Isolated nodes go to a circle of their own if needed
    isolated = [n for n in G.nodes if n not in pos]
    for k, node in enumerate(isolated):
        theta = 2 * np.pi * k / max(1, len(isolated))
        pos[node] = (1.4 * np.cos(theta), 1.4 * np.sin(theta))

    nx.draw_networkx_nodes(G, pos, node_color='#9ecae1', edgecolors='#2171b5', node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#525252', width=1.5, ax=ax)
    ax.set_title('Conflict graph between campaigns')
    ax.set_aspect('equal')
    ax.axis('off')
    _save(fig, save_name)
    
    #return fig


def plot_conflict_graph_between_campaigns_matrix_version(environment, save_name):
    """Plot of the conflict graph in matrix form between campaigns.

    A pleasant blue palette is used; 0/1 conflict indicators are drawn inside
    each cell and the cells are separated by white grid lines.
    """
    cg = environment.conflicts_graph
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = matplotlib.colors.ListedColormap(['#f7fbff', '#4292c6'])
    ax.imshow(cg.graph_matrix, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(cg.n_campaigns))
    ax.set_yticks(range(cg.n_campaigns))
    # separator lines between cells
    ax.set_xticks(np.arange(-0.5, cg.n_campaigns, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cg.n_campaigns, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    # 0/1 indicators inside cells
    for i in range(cg.n_campaigns):
        for j in range(cg.n_campaigns):
            ax.text(j, i, str(int(cg.graph_matrix[i, j])), ha='center', va='center',
                    color='white' if cg.graph_matrix[i, j] else '#525252', fontsize=11)
    ax.set_xlabel('Campaign')
    ax.set_ylabel('Campaign')
    ax.set_title('Conflict graph (matrix)')
    _save(fig, save_name)
    
    #return fig


# ============================================================
# Req 1 specific plots
# ============================================================

def plot_req1_sampled_max_bids_in_time(environment):
    """Plot in time of the maximum bids sampled in the Requirement 1 environment.

    The average max-bid across the rounds is shown as a red dashed horizontal line.
    """
    n_campaigns = environment.N_CAMPAIGNS
    fig, axes = plt.subplots(1, n_campaigns, figsize=(6 * n_campaigns, 4), squeeze=False)
    axes = axes[0]
    for i, campaign in enumerate(environment.campaigns):
        m = campaign.get_max_competing_bids()
        axes[i].plot(m, label='$m_t$')
        axes[i].axhline(np.mean(m), color='red', linestyle='--', label=r'$\bar{m}$')
        axes[i].set_ylim(0, 1)
        axes[i].set_xlabel('$t$')
        axes[i].set_ylabel('$m_t$')
        axes[i].set_title(f'Sampled max bids in time (campaign {i+1})')
        axes[i].legend()
    plt.tight_layout()
    _save(fig, 'req1_max_bids_ts.png')
    
    #return fig


def plot_req1_max_bids_histogram(environment):
    """Histogram of the maximum bids sampled in the Requirement 1 environment.

    The true Beta distribution is also superimposed for comparison.
    """
    n_campaigns = environment.N_CAMPAIGNS
    fig, axes = plt.subplots(1, n_campaigns, figsize=(6 * n_campaigns, 4), squeeze=False)
    axes = axes[0]
    for i, campaign in enumerate(environment.campaigns):
        m = campaign.get_max_competing_bids()
        axes[i].hist(m, bins=50, density=True, alpha=0.6, label='Empirical')
        x = np.linspace(0.0, 1.0, 200)
        pdf = scipy.stats.beta.pdf(x, a=campaign.N_COMPETITORS, b=1)
        axes[i].plot(x, pdf, 'r-', lw=2, label='Beta pdf')
        axes[i].set_xlabel('$m$')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Max bids histogram (campaign {i+1})')
        axes[i].legend()
    plt.tight_layout()
    _save(fig, 'req1_max_bids_hist.png')
    
    #return fig


def plot_req1_clairvoyant_distribution_over_bids(experiment, save_name, bidder_index=-1):
    """Bar chart visualization of the optimal distribution over bids according to the clairvoyant.

    Uses the clairvoyant gammas of the last trial. Only the K-1 non-zero bids are
    plotted (one bar per non-zero bid, with a width large enough to read); the
    0.0 bid is omitted because it has no gamma value.
    """
    bidder = experiment.bidders[bidder_index]
    gammas = experiment.results["clairvoyant_campaign_gammas"][bidder_index]
    bids = bidder.bids  # full bid grid, length K (includes 0.0)
    nonzero_bids = bids[1:]  # the K-1 non-zero bids

    # Coerce to a uniform (N_CAMPAIGNS, K-1) shape (one row per campaign, one column
    # per non-zero bid). The simplified clairvoyant stores gammas over the
    # non-zero bids only, so the campaign gammas have length K-1. The true
    # combinatorial clairvoyant stores a joint distribution over (campaign, bid)
    # of length K, but for the bid-only marginal we use a mean over (phase,
    # superarm) and then drop the 0.0 bid.
    if gammas.ndim == 2:
        # shape (N_CAMPAIGNS, K-1) for the simplified clairvoyant
        per_campaign_gammas = gammas
    elif gammas.ndim == 3:
        # shape (N_CAMPAIGNS, n_phases, K-1) for the simplified clairvoyant with phases
        per_campaign_gammas = gammas.mean(axis=1)
    elif gammas.ndim == 4:
        # shape (n_phases, 2**N_CAMPAIGNS, N_CAMPAIGNS, K) for the true combinatorial
        # Drop the 0.0 bid (last axis index 0) and average over (phase, superarm).
        per_campaign_gammas = gammas[..., 1:].mean(axis=(0, 1))
    else:
        raise ValueError(f"Unsupported gamma ndim: {gammas.ndim}")

    n_campaigns = per_campaign_gammas.shape[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = (nonzero_bids[1] - nonzero_bids[0]) * 0.8 / max(1, n_campaigns)
    x_centers = np.arange(len(nonzero_bids))
    for i in range(n_campaigns):
        # Renormalize the gamma values so they sum to 1 in case the mean over
        # (phase, superarm) does not preserve the simplex constraint.
        values = per_campaign_gammas[i]
        s = values.sum()
        if s > 0:
            values = values / s
        ax.bar(x_centers + (i - (n_campaigns - 1) / 2) * width, values, width=width, label=f'Campaign {i}')
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f'{b:.1f}' for b in nonzero_bids])
    ax.set_xlabel('$b$')
    ax.set_ylabel('$\\gamma(b)$')
    ax.set_title('Clairvoyant distribution over bids')
    ax.legend()
    _save(fig, save_name)
    
    #return fig


def plot_req1_nobudget_vs_budget_regrets_comparison(nobudget_experiment, budget_experiment):
    """Plot of the difference in regrets between the no-budget and budget settings in Requirement 1.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    n_users = nobudget_experiment.n_users
    ax.plot(np.arange(n_users), nobudget_experiment.results["avg_regrets"],
                 label=f'{nobudget_experiment.bidder_class.__name__} (no budget)')
    ax.fill_between(
        np.arange(n_users),
        nobudget_experiment.results["avg_regrets"] - nobudget_experiment.results["std_regrets"],
        nobudget_experiment.results["avg_regrets"] + nobudget_experiment.results["std_regrets"],
        alpha=0.2
    )
    ax.plot(np.arange(n_users), budget_experiment.results["avg_regrets"],
                 label=f'{budget_experiment.bidder_class.__name__} (budget)')
    ax.fill_between(
        np.arange(n_users),
        budget_experiment.results["avg_regrets"] - budget_experiment.results["std_regrets"],
        budget_experiment.results["avg_regrets"] + budget_experiment.results["std_regrets"],
        alpha=0.2
    )
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\\sum R_t$')
    ax.set_title('Regret comparison')
    ax.legend()

    plt.tight_layout()
    _save(fig, 'req1_regret_compare.png')
    
    #return fig


def plot_req1_budget_rho_ratio(budget_experiment):
    """Plot of the difference in regrets between the no-budget and budget settings in Requirement 1.
    Cumulative spend / (rho * t) is shown against 1.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    n_users = budget_experiment.n_users
    rho = budget_experiment.starting_budget / n_users
    total_cumulative_payments_by_trial = budget_experiment.results["total_cumulative_payments_by_trial"]
        
    rho_ratio = total_cumulative_payments_by_trial / (rho * np.arange(1, n_users + 1)[None, :])
    avg_rho_ratio = rho_ratio.mean(axis=0)
    std_rho_ratio = rho_ratio.std(axis=0)
    ax.plot(np.arange(n_users), avg_rho_ratio, label=r'$\sum c_t / (\rho t)$')
    ax.fill_between(
        np.arange(n_users),
        avg_rho_ratio - std_rho_ratio,
        avg_rho_ratio + std_rho_ratio,
        alpha=0.2
    )
    ax.axhline(1.0, color='red', linestyle='--', label=r'$\sum c_t / (\rho t) = 1$')
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\sum c_t / (\rho t)$')
    ax.set_title(r'Budget pacing: $\rho$-ratio')
    ax.legend()

    plt.tight_layout()
    _save(fig, 'req1_rho_ratio.png')
    
    #return fig


def plot_req1_budget_pacing(experiment):
    """Plot of the budget pacing over time in Requirement 1.

    Shows the budget spent until now and the expected total budget spend according
    to the per-round budget rho. The expected per-round budget rho is shown as a
    red dashed line for comparison. A budget-exhausted vertical line is drawn
    when the average cumulative spend has stopped growing.
    """
    results = experiment.results
    n_users = experiment.n_users
    rho = experiment.starting_budget / n_users
    avg_total_cumulative_payments = results["avg_total_cumulative_payments"]
    std_total_cumulative_payments = results["std_total_cumulative_payments"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(n_users), avg_total_cumulative_payments, label='Cumulative spend')
    ax.fill_between(
        np.arange(n_users),
        avg_total_cumulative_payments - std_total_cumulative_payments,
        avg_total_cumulative_payments + std_total_cumulative_payments,
        alpha=0.2
    )
    ax.plot(np.arange(n_users), rho * np.arange(1, n_users + 1), 'r--', label=r'$\rho \cdot t$')
    budget_exhausted_t = _budget_exhausted_t_from_avg_payments(experiment)
    if budget_exhausted_t is not None and budget_exhausted_t < n_users - 1:
        ax.axvline(budget_exhausted_t, color='green', linestyle=':', label='Budget exhausted')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\\sum c_t$')
    ax.set_title('Budget pacing')
    ax.legend()
    _save(fig, 'req1_budget_pacing.png')

    #return fig


def plot_req1_per_round_payment_vs_rho(experiment):
    """Per-round payment vs rho for Requirement 1."""
    plot_per_round_payment_vs_rho(experiment, 'req1_per_round_payment_vs_rho.png')

    #return fig


# ============================================================
# Req 2 specific plots
# ============================================================

def plot_req2_sampled_max_bids_in_time(environment):
    """Plot in time of the maximum bids sampled in the Requirement 2 environment.

    The average max-bid across the rounds is shown as a red dashed horizontal line.
    One figure with multiple plots, one for each campaign.
    """
    n_campaigns = environment.N_CAMPAIGNS
    n_cols = min(3, n_campaigns)
    n_rows = (n_campaigns + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for i, campaign in enumerate(environment.campaigns):
        r, c = i // n_cols, i % n_cols
        m = campaign.get_max_competing_bids()
        axes[r][c].plot(m, label='$m_t$')
        axes[r][c].axhline(np.mean(m), color='red', linestyle='--', label=r'$\bar{m}$')
        axes[r][c].set_ylim(0, 1)
        axes[r][c].set_xlabel('$t$')
        axes[r][c].set_ylabel('$m_t$')
        axes[r][c].set_title(f'Sampled max bids (campaign {i+1})')
        axes[r][c].legend()
    for j in range(n_campaigns, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r][c].axis('off')
    plt.tight_layout()
    _save(fig, 'req2_max_bids_ts.png')
    
    #return fig


def plot_req2_max_bids_histogram(environment):
    """Histogram of the maximum bids sampled in the Requirement 2 environment.

    The true Beta distribution is also superimposed for comparison.
    One figure with multiple plots, one for each campaign.
    """
    n_campaigns = environment.N_CAMPAIGNS
    n_cols = min(3, n_campaigns)
    n_rows = (n_campaigns + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for i, campaign in enumerate(environment.campaigns):
        r, c = i // n_cols, i % n_cols
        m = campaign.get_max_competing_bids()
        axes[r][c].hist(m, bins=50, density=True, alpha=0.6, label='Empirical')
        x = np.linspace(0.0, 1.0, 200)
        pdf = scipy.stats.beta.pdf(x, a=campaign.N_COMPETITORS, b=1)
        axes[r][c].plot(x, pdf, 'r-', lw=2, label='Beta pdf')
        axes[r][c].set_xlabel('$m$')
        axes[r][c].set_ylabel('Density')
        axes[r][c].set_title(f'Max bids histogram (campaign {i+1})')
        axes[r][c].legend()
    for j in range(n_campaigns, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r][c].axis('off')
    plt.tight_layout()
    _save(fig, 'req2_max_bids_hist.png')
    
    #return fig


def plot_req2_clairvoyant_distribution_over_bids(experiment=None, bidder=None, gammas=None, bidder_index=-1):
    """Bar chart visualization of the optimal distribution over bids according to the clairvoyant.

    Uses the last trial's clairvoyant gammas. Handles the simplified shape
    (N_CAMPAIGNS, K-1) and the true shape (n_phases, 2**N_CAMPAIGNS, N_CAMPAIGNS, K).
    Only the K-1 non-zero bids are plotted; per-campaign gammas are normalized
    to sum to 1.
    """
    if experiment is not None and bidder is not None:
        raise ValueError("Provide either experiment or both bidder and gammas, not both.")
    if experiment is not None:
        bidder = experiment.bidders[bidder_index]
        gammas = experiment.results["clairvoyant_campaign_gammas"][bidder_index]
    elif bidder is not None and gammas is not None:
        pass
    else:
        raise ValueError("Either experiment or both bidder and gammas must be provided.")

    bids = bidder.bids  # length K
    nonzero_bids = bids[1:]  # length K-1

    if gammas.ndim == 3:
        per_campaign_gammas = gammas.mean(axis=1)
    elif gammas.ndim == 4:
        # Drop the 0.0 bid, average over (phase, superarm)
        per_campaign_gammas = gammas[..., 1:].mean(axis=(0, 1))
    else:
        raise ValueError(f"Unsupported gamma ndim: {gammas.ndim}")

    n_campaigns = per_campaign_gammas.shape[0]

    n_cols = min(3, n_campaigns)
    n_rows = (n_campaigns + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    width = (nonzero_bids[1] - nonzero_bids[0]) * 0.8
    x_centers = np.arange(len(nonzero_bids))
    for i in range(n_campaigns):
        r, c = i // n_cols, i % n_cols
        ax = axes[r][c]
        values = per_campaign_gammas[i]
        s = values.sum()
        if s > 0:
            values = values / s
        ax.bar(x_centers, values, width=width)
        ax.set_xticks(x_centers)
        ax.set_xticklabels([f'{b:.1f}' for b in nonzero_bids])
        ax.set_xlabel('$b$')
        ax.set_ylabel('$\\gamma(b)$')
        ax.set_title(f'Clairvoyant $\\gamma$ (campaign {i+1})')
    for j in range(n_campaigns, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r][c].axis('off')
    plt.tight_layout()
    _save(fig, 'req2_clairvoyant_gamma_per_campaign.png')
    
    #return fig


def plot_req2_clairvoyant_distribution_over_campaigns_subsets(save_name, experiment=None, superarm_gammas=None, bidder_index=-1):
    """Bar chart visualization of the optimal distribution over superarms according to the clairvoyant.

    Used with the "true" combinatorial clairvoyant.
    """
    if experiment is not None and superarm_gammas is not None:
        raise ValueError("Provide either experiment or superarm_gammas, not both.")
    if experiment is not None:
        superarm_gammas = experiment.results["clairvoyant_superarm_gammas"][bidder_index]
    elif superarm_gammas is not None:
        pass
    else:
        raise ValueError("Either experiment or superaR  gammas must be provided.")
    
    n_campaigns = np.log2(superarm_gammas.shape[-1]).astype(int)
    
    labels = [format(a, f'0{n_campaigns}b') for a in range(superarm_gammas.shape[-1])]
    fig, ax = plt.subplots(figsize=(10, 5))
    if superarm_gammas.ndim == 2:
        avg = superarm_gammas.mean(axis=0)  # average over phases
    else:
        avg = superarm_gammas
    ax.bar(labels, avg)
    ax.set_xlabel('Campaign subset (binary)')
    ax.set_ylabel('$\\pi(a)$')
    ax.set_title('Clairvoyant distribution over superarms')
    plt.xticks(rotation=90)
    plt.tight_layout()
    _save(fig, save_name)
    
    #return fig


def plot_req2_trend_total_LP_variables_growing_with_n_campaigns_and_n_bids(
        starting_n_bids=2, starting_n_campaigns=2, step=1, n_steps=6):
    """Plot of the trend of the total number of LP variables growing with n_campaigns and n_bids.

    Three lines:
    - n_bids^n_campaigns
    - true clairvoyant = 2^n_campaigns * n_bids * n_campaigns + 2^n_campaigns
    - simplified clairvoyant = n_campaigns * n_bids + 2^n_campaigns
    """
    nb = np.arange(starting_n_bids, starting_n_bids + n_steps * step, step)
    nc = np.arange(starting_n_campaigns, starting_n_campaigns + n_steps * step, step)
    pairs = list(zip(nb, nc))
    x = np.arange(len(pairs))
    naive = np.array([b ** c for b, c in pairs])
    true = np.array([(2 ** c) * b * c + (2 ** c) for b, c in pairs])
    simplified = np.array([c * b + (2 ** c) for b, c in pairs])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, naive, 'o-', label='$n_{\\mathrm{bids}}^{n_{\\mathrm{campaigns}}}$')
    ax.plot(x, true, 's-', label='True clairvoyant')
    ax.plot(x, simplified, '^-', label='Simplified clairvoyant')
    ax.set_xticks(x)
    ax.set_xticklabels([f'({b},{c})' for b, c in pairs])
    ax.set_xlabel('$(n_{\\mathrm{bids}}, n_{\\mathrm{campaigns}})$')
    ax.set_ylabel('# LP variables')
    ax.set_yscale('log')
    ax.set_title('Trend of total LP variables')
    ax.legend()
    plt.tight_layout()
    _save(fig, 'req2_lp_variables_trend.png')
    
    #return fig


def plot_req2_combinatorial_simplified_vs_combinatorial_true_regrets_comparison(
        simplified_experiment, true_experiment):
    """Plot of the difference in regrets between the CombinatorialUCBSimplified and CombinatorialUCBTrue bidders."""
    n_users = simplified_experiment.n_users
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(np.arange(n_users),
                 simplified_experiment.results["avg_regrets"],
                 label='Simplified')
    axes[0].fill_between(
        np.arange(n_users),
        simplified_experiment.results["avg_regrets"] - simplified_experiment.results["std_regrets"],
        simplified_experiment.results["avg_regrets"] + simplified_experiment.results["std_regrets"],
        alpha=0.2
    )
    axes[0].plot(np.arange(n_users),
                 true_experiment.results["avg_regrets"],
                 label='True')
    axes[0].fill_between(
        np.arange(n_users),
        true_experiment.results["avg_regrets"] - true_experiment.results["std_regrets"],
        true_experiment.results["avg_regrets"] + true_experiment.results["std_regrets"],
        alpha=0.2
    )
    axes[0].set_xlabel('$t$')
    axes[0].set_ylabel('$\\sum R_t$')
    axes[0].set_title('Simplified vs True: cumulative regret')
    axes[0].legend()

    diff = simplified_experiment.results["avg_regrets"] - true_experiment.results["avg_regrets"]
    axes[1].plot(np.arange(n_users), diff, label='Simplified - True')
    axes[1].axhline(0.0, color='red', linestyle='--')
    axes[1].set_xlabel('$t$')
    axes[1].set_ylabel('Regret difference')
    axes[1].set_title('Regret difference (Simplified - True)')
    axes[1].legend()
    plt.tight_layout()
    _save(fig, 'req2_simplified_vs_true_regret.png')
    
    #return fig


def plot_req2_budget_pacing(experiment):
    """Plot of the budget pacing over time in Requirement 2.

    Both the total budget and the per-campaign budgets are shown in the same plot,
    with an appropriate legend. The expected per-round budget rho is shown as a red
    dashed line for comparison. The budget-exhausted line is also drawn.
    """
    results = experiment.results
    n_users = experiment.n_users
    rho = experiment.starting_budget / n_users
    avg_total_cumulative_payments = results["avg_total_cumulative_payments"]
    std_total_cumulative_payments = results["std_total_cumulative_payments"]
    n_campaigns = experiment.n_campaigns
    budget_exhausted_t = _budget_exhausted_t_from_avg_payments(experiment)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(n_users), avg_total_cumulative_payments, color='black', lw=2, label='Total cumulative spend')
    ax.fill_between(
        np.arange(n_users),
        avg_total_cumulative_payments - std_total_cumulative_payments,
        avg_total_cumulative_payments + std_total_cumulative_payments,
        alpha=0.2, color='black'
    )
    #for i in range(n_campaigns):
    #    ax.plot(np.arange(n_users), results["avg_payments"][:, i], label=f'Campaign {i}')
    ax.plot(np.arange(n_users), rho * np.arange(1, n_users + 1), 'r--', label=r'$\rho \cdot t$')
    if budget_exhausted_t is not None and budget_exhausted_t < n_users - 1:
        ax.axvline(budget_exhausted_t, color='green', linestyle=':', label='Budget exhausted')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\\sum c_t$')
    ax.set_title('Budget pacing')
    ax.legend()
    _save(fig, 'req2_budget_pacing.png')

    #return fig


def plot_req2_per_round_payment_vs_rho(experiment):
    """Per-round payment vs rho for Requirement 2."""
    plot_per_round_payment_vs_rho(experiment, 'req2_per_round_payment_vs_rho.png')

    #return fig


# ============================================================
# Req 3 specific plots
# ============================================================

def plot_req3_sampled_max_bids_in_time(environment):
    """Plot in time of the maximum bids sampled in the Requirement 3 environment.

    The average max-bid across the rounds is shown as a red dashed horizontal line.
    One figure with multiple plots, one for each campaign.
    """
    n_campaigns = environment.N_CAMPAIGNS
    n_cols = min(3, n_campaigns)
    n_rows = (n_campaigns + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for i, campaign in enumerate(environment.campaigns):
        r, c = i // n_cols, i % n_cols
        m = campaign.get_max_competing_bids()
        axes[r][c].plot(m, label='$m_t$')
        axes[r][c].axhline(np.mean(m), color='red', linestyle='--', label=r'$\bar{m}$')
        axes[r][c].set_ylim(0, 1)
        axes[r][c].set_xlabel('$t$')
        axes[r][c].set_ylabel('$m_t$')
        axes[r][c].set_title(f'Sampled max bids (campaign {i+1})')
        axes[r][c].legend()
    for j in range(n_campaigns, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r][c].axis('off')
    plt.tight_layout()
    _save(fig, 'req3_max_bids_ts.png')
    
    #return fig


def plot_req3_max_bids_histogram(environment):
    """Histogram of the maximum bids sampled in the Requirement 3 environment.

    One figure with multiple plots, one for each campaign.
    """
    n_campaigns = environment.N_CAMPAIGNS
    n_cols = min(3, n_campaigns)
    n_rows = (n_campaigns + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for i, campaign in enumerate(environment.campaigns):
        r, c = i // n_cols, i % n_cols
        m = campaign.get_max_competing_bids()
        axes[r][c].hist(m, bins=50, density=True, alpha=0.6, label='Empirical')
        axes[r][c].set_xlabel('$m$')
        axes[r][c].set_ylabel('Density')
        axes[r][c].set_title(f'Max bids histogram (campaign {i+1})')
        axes[r][c].legend()
    for j in range(n_campaigns, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r][c].axis('off')
    plt.tight_layout()
    _save(fig, 'req3_max_bids_hist.png')
    
    #return fig


def plot_req3_clairvoyant_distribution_over_campaigns_subsets(experiment=None, superarm_gammas=None, bidder_index=-1):
    """Bar chart visualization of the optimal distribution over superarms according to the clairvoyant.

    Used with the "true" combinatorial clairvoyant.
    """
    if experiment is not None and superarm_gammas is not None:
        raise ValueError("Provide either experiment or superarm_gammas, not both.")
    if experiment is not None:
        superarm_gammas = experiment.results["clairvoyant_superarm_gammas"][bidder_index]
    elif superarm_gammas is not None:
        pass
    else:
        raise ValueError("Either experiment or superaR  gammas must be provided.")
    
    n_campaigns = np.log2(superarm_gammas.shape[-1]).astype(int)
    
    labels = [format(a, f'0{n_campaigns}b') for a in range(superarm_gammas.shape[-1])]
    fig, ax = plt.subplots(figsize=(10, 5))
    if superarm_gammas.ndim == 2:
        avg = superarm_gammas.mean(axis=0)  # average over phases
    else:
        avg = superarm_gammas
    ax.bar(labels, avg)
    ax.set_xlabel('Campaign subset (binary)')
    ax.set_ylabel('$\\pi(a)$')
    ax.set_title('Clairvoyant distribution over superarms')
    plt.xticks(rotation=90)
    plt.tight_layout()
    _save(fig, 'req3_clairvoyant_superarm.png')
    
    #return fig


def plot_req3_budget_pacing(experiment):
    """Plot of the budget pacing over time in Requirement 3.

    Both the total budget and the per-campaign budgets are shown in the same plot,
    with an appropriate legend. The expected per-round budget rho is shown as a red
    dashed line for comparison. The budget-exhausted line is also drawn.
    """
    results = experiment.results
    n_users = experiment.n_users
    rho = experiment.starting_budget / n_users
    avg_total_cumulative_payments = results["avg_total_cumulative_payments"]
    std_total_cumulative_payments = results["std_total_cumulative_payments"]
    n_campaigns = experiment.n_campaigns
    budget_exhausted_t = _budget_exhausted_t_from_avg_payments(experiment)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(n_users), avg_total_cumulative_payments, color='black', lw=2, label='Total cumulative spend')
    ax.fill_between(
        np.arange(n_users),
        avg_total_cumulative_payments - std_total_cumulative_payments,
        avg_total_cumulative_payments + std_total_cumulative_payments,
        alpha=0.2, color='black'
    )
    #for i in range(n_campaigns):
    #    ax.plot(np.arange(n_users), results["avg_payments"][:, i], label=f'Campaign {i}')
    ax.plot(np.arange(n_users), rho * np.arange(1, n_users + 1), 'r--', label=r'$\rho \cdot t$')
    if budget_exhausted_t is not None and budget_exhausted_t < n_users - 1:
        ax.axvline(budget_exhausted_t, color='green', linestyle=':', label='Budget exhausted')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\\sum c_t$')
    ax.set_title('Budget pacing')
    ax.legend()
    _save(fig, 'req3_budget_pacing.png')

    #return fig


def plot_req3_per_round_payment_vs_rho(experiment):
    """Per-round payment vs rho for Requirement 3."""
    plot_per_round_payment_vs_rho(experiment, 'req3_per_round_payment_vs_rho.png')

    #return fig


def plot_req3_dual_variable_lambda_over_time(experiment):
    """Plot of the dual variable lambda over time in Requirement 3.

    Uses the `lambda_history` attribute of the bidder from the last trial.
    """
    bidder = experiment.bidders[-1]
    if not hasattr(bidder, 'lambda_history') or len(bidder.lambda_history) == 0:
        raise ValueError("Bidder does not expose a non-empty lambda_history")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(bidder.lambda_history)), bidder.lambda_history, label=r'$\lambda_t$')
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\lambda_t$')
    ax.set_title('Dual variable lambda')
    ax.legend()
    _save(fig, 'req3_dual_lambda.png')
    
    #return fig

# ============================================================
# Req 4 specific plots
# ============================================================

def plot_req4_sampled_max_bids_in_time(environment):
    """Plot in time of the maximum bids sampled in the Requirement 4 environment.

    The phase change times are also shown as vertical dashed lines, and the
    per-phase average max-bid is drawn as a horizontal red dashed line that
    switches at each phase change (rendered as a discontinuous step).
    One figure with multiple plots, one for each campaign.
    """
    n_campaigns = environment.N_CAMPAIGNS
    n_cols = min(3, n_campaigns)
    n_rows = (n_campaigns + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for i, campaign in enumerate(environment.campaigns):
        r, c = i // n_cols, i % n_cols
        m = campaign.get_max_competing_bids()
        axes[r][c].plot(m, label='$m_t$')
        # Phase boundaries: [0, t1, t2, ..., tn, T] used to slice the signal into phases.
        phase_bounds = [0] + list(campaign.phase_change_times) + [len(m)]
        for k in range(len(phase_bounds) - 1):
            t_start, t_end = phase_bounds[k], phase_bounds[k + 1]
            if t_start == t_end:
                continue
            phase_mean = np.mean(m[t_start:t_end])
            # Draw the horizontal line only across its phase, so it appears discontinuous.
            axes[r][c].hlines(phase_mean, t_start, t_end, color='red', linestyle='--', alpha=0.8,
                              label=r'$\bar{m}$' if (k == 0) else None)
            axes[r][c].axvline(t_end, color='red', linestyle='--', alpha=0.4)
        axes[r][c].set_ylim(0, 1)
        axes[r][c].set_xlabel('$t$')
        axes[r][c].set_ylabel('$m_t$')
        axes[r][c].set_title(f'Sampled max bids (campaign {i+1})')
        axes[r][c].legend()
    for j in range(n_campaigns, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r][c].axis('off')
    plt.tight_layout()
    _save(fig, 'req4_max_bids_ts_with_changes.png')
    
    #return fig


def plot_req4_clairvoyant_distribution_over_campaigns_subsets(experiment=None, superarm_gammas=None, bidder_index=-1):
    """Bar chart visualization of the optimal distribution over superarms according to the clairvoyant.

    Used with the "true" combinatorial clairvoyant.
    """
    if experiment is not None and superarm_gammas is not None:
        raise ValueError("Provide either experiment or superarm_gammas, not both.")
    if experiment is not None:
        superarm_gammas = experiment.results["clairvoyant_superarm_gammas"][bidder_index]
    elif superarm_gammas is not None:
        pass
    else:
        raise ValueError("Either experiment or superaR  gammas must be provided.")
    
    n_campaigns = np.log2(superarm_gammas.shape[-1]).astype(int)
    n_phases = superarm_gammas.shape[0]
    
    labels = [format(a, f'0{n_campaigns}b') for a in range(superarm_gammas.shape[-1])]

    n_cols = min(4, n_phases)
    n_rows = (n_phases + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for i, gammas in enumerate(superarm_gammas):
        r, c = i // n_cols, i % n_cols
        axes[r][c].bar(labels, gammas)
        axes[r][c].set_ylim(0, 1)
        axes[r][c].set_xlabel('Campaign subset (binary)')
        axes[r][c].set_ylabel('$\\pi(a)$')
        axes[r][c].set_title(f'Clairvoyant distribution over superarms (phase {i})')
        axes[r][c].tick_params(axis='x', rotation=90)
    for j in range(n_phases, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r][c].axis('off')

    plt.tight_layout()
    _save(fig, 'req4_clairvoyant_gamma_phases.png')
    
    #return fig


def plot_req4_cusum_cumulative_regret_phase_changes(environment, cusum_experiment):
    """Plot of the cumulative regret over time in Requirement 4.

    Vertical dashed lines indicate the phase change times; vertical dotted lines of
    another color indicate the times in which the CUSUM change detector detected a
    change. The regret is averaged across trials (mean +/- std band). The CUSUM
    reset times are taken from the last trial only.
    """
    n_users = cusum_experiment.n_users
    avg_regret = np.asarray(cusum_experiment.results["avg_regrets"]).flatten()
    std_regret = np.asarray(cusum_experiment.results["std_regrets"]).flatten()

    cusum_resets = []
    bidder = cusum_experiment.bidders[-1]
    if hasattr(bidder, 'reset_history'):
        for campaign in bidder.reset_history:
            for arm_resets in campaign:
                cusum_resets.extend(arm_resets)
    cusum_resets = sorted(set(int(t) for t in cusum_resets))

    phase_times = _all_phase_change_times(environment)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(n_users), avg_regret, label='Cumulative regret (avg)')
    ax.fill_between(
        np.arange(n_users),
        avg_regret - std_regret,
        avg_regret + std_regret,
        alpha=0.2, label='Cumulative regret (std)'
    )
    for i, t in enumerate(phase_times):
        ax.axvline(t, color='red', linestyle='--', alpha=0.6,
                   label='Phase change' if i == 0 else None)
    for i, t in enumerate(cusum_resets):
        ax.axvline(t, color='blue', linestyle=':', alpha=0.6,
                   label='CUSUM reset' if i == 0 else None)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\\sum R_t$')
    ax.set_title('Cumulative regret with CUSUM resets')
    ax.legend()
    plt.tight_layout()
    _save(fig, 'req4_cusum_resets.png')

    #return fig


def plot_req4_cumulative_regrets_combined(environment, experiments_by_bidder_name):
    """Plot of the cumulative regret over time in Requirement 4 for several bidders.

    Uses the average cumulative regret (mean +/- std band) across trials for each
    bidder. Phase change times are shown as vertical dashed lines.
    """
    n_users = next(iter(experiments_by_bidder_name.values())).n_users
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, experiment in experiments_by_bidder_name.items():
        avg_regret = np.asarray(experiment.results["avg_regrets"]).flatten()
        std_regret = np.asarray(experiment.results["std_regrets"]).flatten()
        ax.plot(np.arange(n_users), avg_regret, label=f'{name} (avg)')
        ax.fill_between(
            np.arange(n_users),
            avg_regret - std_regret,
            avg_regret + std_regret,
            alpha=0.2
        )
    for t in _all_phase_change_times(environment):
        ax.axvline(t, color='red', linestyle='--', alpha=0.6)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\\sum R_t$')
    ax.set_title('Cumulative regret comparison')
    ax.legend()
    plt.tight_layout()
    _save(fig, 'req4_regret_comparison.png')

    #return fig


def plot_req4_budget_pacing(experiments_by_bidder_name):
    """Plot of the budget pacing over time for all bidders (req4).

    Only the total budget of each bidder is shown in the same plot, with the
    budget-exhausted line of each bidder drawn at the moment its average
    cumulative spend has stopped growing.
    """
    n_users = next(iter(experiments_by_bidder_name.values())).n_users
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, experiment in experiments_by_bidder_name.items():
        avg_total_cumulative_payments = experiment.results["avg_total_cumulative_payments"]
        std_total_cumulative_payments = experiment.results["std_total_cumulative_payments"]
        ax.plot(np.arange(n_users), avg_total_cumulative_payments, label=f'{name} (total)')
        ax.fill_between(
            np.arange(n_users),
            avg_total_cumulative_payments - std_total_cumulative_payments,
            avg_total_cumulative_payments + std_total_cumulative_payments,
            alpha=0.2
        )
        budget_exhausted_t = _budget_exhausted_t_from_avg_payments(experiment)
        if budget_exhausted_t is not None and budget_exhausted_t < n_users - 1:
            ax.axvline(budget_exhausted_t, color='green', linestyle=':', alpha=0.5)
    rho = next(iter(experiments_by_bidder_name.values())).starting_budget / n_users
    ax.plot(np.arange(n_users), rho * np.arange(1, n_users + 1), 'r--', label=r'$\rho \cdot t$')
    for name, experiment in experiments_by_bidder_name.items():
        budget_exhausted_t = _budget_exhausted_t_from_avg_payments(experiment)
        if budget_exhausted_t is not None and budget_exhausted_t < n_users - 1:
            ax.axvline(budget_exhausted_t, color='green', linestyle=':', alpha=0.5,
                       label=f'{name} budget exhausted' if name == next(iter(experiments_by_bidder_name)) else None)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\\sum c_t$')
    ax.set_title('Budget pacing comparison')
    ax.legend()
    plt.tight_layout()
    _save(fig, 'req4_budget_pacing_compare.png')

    #return fig


def plot_req4_per_round_payment_vs_rho(experiments_by_bidder_name):
    """Per-round payment c_t vs rho for all bidders in Requirement 4.

    Each bidder's total per-round payment (summed across campaigns) is plotted on
    the same axes with mean +/- std band. The per-round budget rho is drawn as a
    red dashed horizontal line: above the line = overspending within a round,
    below = underspending.
    """
    n_users = next(iter(experiments_by_bidder_name.values())).n_users
    rho = next(iter(experiments_by_bidder_name.values())).starting_budget / n_users

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, experiment in experiments_by_bidder_name.items():
        per_round = np.array(experiment.results["my_payments"]).sum(axis=-1)      # shape (n_trials, n_users)
        avg = per_round.mean(axis=0)
        std = per_round.std(axis=0)
        ax.plot(np.arange(n_users), avg, label=f'{name} (avg)')
        ax.fill_between(
            np.arange(n_users),
            avg - std,
            avg + std,
            alpha=0.2
        )
    ax.axhline(rho, color='red', linestyle='--', label=r'$\rho$')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$c_t$')
    ax.set_title('Per-round payment vs $\\rho$')
    ax.legend()
    plt.tight_layout()
    _save(fig, 'req4_per_round_payment_vs_rho.png')

    #return fig


def plot_req4_dual_variable_lambda_over_time(experiments_by_bidder_name):
    """Plot of the dual variable lambda over time for the bidders that have one.

    Uses the `lambda_history` from the last trial of each bidder.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for name, experiment in experiments_by_bidder_name.items():
        bidder = experiment.bidders[-1]
        if hasattr(bidder, 'lambda_history') and len(bidder.lambda_history) > 0:
            ax.plot(np.arange(len(bidder.lambda_history)), bidder.lambda_history, label=name)
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, 'No bidder exposes lambda_history', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\lambda_t$')
    ax.set_title('Dual variable lambda')
    ax.legend()
    plt.tight_layout()
    _save(fig, 'req4_dual_lambda_compare.png')
    
    #return fig