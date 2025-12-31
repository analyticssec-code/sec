import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components


# ----------------------------
# Mock data generation
# ----------------------------
def _make_username(rng: random.Random) -> str:
    bases = ["coolcat", "dragon", "pokerking", "sunrise", "nightowl", "fastfish", "silent", "maxbet", "river", "ace"]
    suffix = rng.randint(10, 9999)
    # Introduce some similarity patterns
    base = rng.choice(bases)
    if rng.random() < 0.15:
        base = base.replace("o", "0").replace("i", "1")
    if rng.random() < 0.10:
        base = base + rng.choice(["_", ".", "-"]) + rng.choice(["x", "xx", "pro", "01"])
    return f"{base}{suffix}"


def generate_mock_data(seed: int = 7):
    rng = random.Random(seed)
    np.random.seed(seed)

    group_cases = [
        ("club_42_community_A", pd.to_datetime("2025-12-22").date()),
        ("club_99_community_B", pd.to_datetime("2025-12-29").date()),
    ]

    all_players_rows = []
    all_edges_rows = []

    for group_id, week_start in group_cases:
        group_case_id = f"{group_id}__{week_start.isoformat()}"
        n_players = rng.randint(22, 35)

        # Create player ids
        puids = [f"P{rng.randint(100000, 999999)}" for _ in range(n_players)]

        # Create some "infra" pools to force shared-device/shared-ip clusters
        device_pool = [f"DEV_{rng.randint(1000, 9999)}" for _ in range(rng.randint(6, 10))]
        ip_pool = [f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}" for _ in range(rng.randint(8, 14))]

        # Assign each player a primary device & IP
        primary_device = {p: rng.choice(device_pool) for p in puids}
        primary_ip = {p: rng.choice(ip_pool) for p in puids}
        username = {p: _make_username(rng) for p in puids}

        # Choose suspicious-from-file and agent-marked players
        suspicious_players = set(rng.sample(puids, k=max(6, n_players // 4)))
        marked_players = set(rng.sample(list(suspicious_players), k=max(3, len(suspicious_players) // 2)))

        # Create player summary rows (nodes table equivalent)
        for p in puids:
            bot_score_ml = float(np.clip(np.random.normal(loc=0.45, scale=0.25), 0, 1))
            # Bias suspicious players a bit higher
            if p in suspicious_players:
                bot_score_ml = float(np.clip(bot_score_ml + np.random.uniform(0.15, 0.35), 0, 1))

            adl_score = int(np.clip(np.random.normal(loc=45, scale=18), 0, 100))
            if p in suspicious_players and rng.random() < 0.35:
                adl_score = int(np.clip(adl_score + rng.randint(15, 35), 0, 100))

            suspicious_hand_count = rng.randint(2, 14) if p in suspicious_players else 0
            marked_hand_count = rng.randint(1, min(8, suspicious_hand_count)) if p in marked_players else 0

            status = "active"
            if bot_score_ml > 0.85 and rng.random() < 0.35:
                status = "banned"
            elif bot_score_ml > 0.70 and rng.random() < 0.35:
                status = "watched"

            all_players_rows.append(
                {
                    "group_case_id": group_case_id,
                    "group_id": group_id,
                    "case_week_start_date": week_start,
                    "platform_uid": p,
                    "username": username[p],
                    "primary_device": primary_device[p],
                    "primary_ip": primary_ip[p],
                    "suspicious_hand_count": suspicious_hand_count,
                    "marked_hand_count": marked_hand_count,
                    "bot_score_ml": bot_score_ml,
                    "adl_score": adl_score,
                    "status": status,
                    "last_seen_ts": datetime(2025, 12, 31) - timedelta(days=rng.randint(0, 15)),
                }
            )

        # Build "context relations" edges from suspicious/marked players out to others
        def add_edge(src, dst, relation_type, strength, evidence_sample, evidence_count):
            if src == dst:
                return
            all_edges_rows.append(
                {
                    "group_case_id": group_case_id,
                    "src_platform_uid": src,
                    "related_platform_uid": dst,
                    "relation_type": relation_type,
                    "relation_strength": float(strength),
                    "evidence_count": int(evidence_count),
                    "evidence_sample": str(evidence_sample),
                    "computed_ts": datetime(2025, 12, 31),
                }
            )

        # Create edges
        for src in suspicious_players:
            # Same device edges
            same_dev = [p for p in puids if p != src and primary_device[p] == primary_device[src]]
            for dst in rng.sample(same_dev, k=min(len(same_dev), rng.randint(1, 4))):
                strength = np.clip(np.random.uniform(0.6, 1.0), 0, 1)
                add_edge(src, dst, "same_primary_device", strength, primary_device[src], rng.randint(5, 60))

            # Same IP edges (noisier)
            same_ip = [p for p in puids if p != src and primary_ip[p] == primary_ip[src]]
            for dst in rng.sample(same_ip, k=min(len(same_ip), rng.randint(0, 3))):
                strength = np.clip(np.random.uniform(0.35, 0.85), 0, 1)
                add_edge(src, dst, "same_primary_ip", strength, primary_ip[src], rng.randint(3, 40))

            # Similar username edges (synthetic heuristic)
            # If usernames share a prefix chunk, connect them
            src_u = username[src]
            candidates = []
            for p in puids:
                if p == src:
                    continue
                u = username[p]
                # crude similarity: common leading chars
                common = 0
                for a, b in zip(src_u, u):
                    if a == b:
                        common += 1
                    else:
                        break
                if common >= 4:
                    candidates.append(p)

            for dst in rng.sample(candidates, k=min(len(candidates), rng.randint(0, 3))):
                strength = np.clip(np.random.uniform(0.45, 0.9), 0, 1)
                add_edge(src, dst, "similar_username", strength, f"{src_u} ~ {username[dst]}", rng.randint(1, 6))

            # High-risk ADL link (connect suspicious to other high ADL players)
            if rng.random() < 0.6:
                high_adl = [p for p in puids if p != src and any(row["adl_score"] >= 60 and row["platform_uid"] == p and row["group_case_id"] == group_case_id
                                                            for row in all_players_rows)]
                for dst in rng.sample(high_adl, k=min(len(high_adl), rng.randint(0, 2))):
                    strength = np.clip(np.random.uniform(0.5, 0.95), 0, 1)
                    add_edge(src, dst, "high_risk_adl", strength, "adl_score>=60", rng.randint(1, 3))

        # Optional: played_together edges among suspicious players (makes clusters more obvious)
        suspicious_list = list(suspicious_players)
        for _ in range(rng.randint(10, 25)):
            a, b = rng.sample(suspicious_list, 2)
            strength = np.clip(np.random.uniform(0.25, 0.9), 0, 1)
            add_edge(a, b, "played_together", strength, f"hands~{rng.randint(20, 250)}", rng.randint(20, 250))

    players_df = pd.DataFrame(all_players_rows)
    edges_df = pd.DataFrame(all_edges_rows)

    # Ensure uniqueness-ish for demo (keep max strength if duplicates)
    if not edges_df.empty:
        edges_df = (
            edges_df.sort_values("relation_strength", ascending=False)
            .drop_duplicates(["group_case_id", "src_platform_uid", "related_platform_uid", "relation_type"])
            .reset_index(drop=True)
        )

    return players_df, edges_df


# ----------------------------
# Graph rendering
# ----------------------------
def build_network_html(players: pd.DataFrame, edges: pd.DataFrame, height_px: int = 720) -> str:
    """
    Build an interactive force network graph (PyVis) HTML.
    Nodes are players; edges are relations.
    """
    net = Network(height=f"{height_px}px", width="100%", directed=False, notebook=False, cdn_resources="in_line")
    net.force_atlas_2based(gravity=-25, central_gravity=0.01, spring_length=120, spring_strength=0.08, damping=0.4)

    # Node sizing: bot_score_ml + suspicious_hand_count
    def node_size(row):
        # size roughly 10..45
        return float(10 + 30 * row["bot_score_ml"] + 1.5 * min(row["suspicious_hand_count"], 10))

    # Simple status to color map (optional; if you prefer default colors, remove color=...)
    status_color = {
        "active": "#7f8c8d",
        "watched": "#f39c12",
        "banned": "#e74c3c",
    }

    # Add nodes
    for _, r in players.iterrows():
        # Format tooltip with newlines (PyVis will render HTML properly)
        title = (
            f"{r['platform_uid']}\n"
            f"Username: {r['username']}\n"
            f"Status: {r['status']}\n"
            f"Bot Score ML: {r['bot_score_ml']:.2f}\n"
            f"ADL Score: {int(r['adl_score'])}\n"
            f"Suspicious Hands: {int(r['suspicious_hand_count'])}\n"
            f"Marked Hands: {int(r['marked_hand_count'])}\n"
            f"Device: {r['primary_device']}\n"
            f"IP: {r['primary_ip']}"
        )

        net.add_node(
            r["platform_uid"],
            label=r["platform_uid"],
            size=node_size(r),
            title=title,
            color=status_color.get(r["status"], "#7f8c8d"),
        )

    # Add edges
    # Edge width based on relation_strength (now uses boosted strength for multi-type edges)
    for _, e in edges.iterrows():
        w = float(1 + 6 * e["relation_strength"])
        # Handle combined relation types (multiple types separated by comma)
        relation_types = e['relation_type']
        num_types = e.get('num_relation_types', 1)
        
        if num_types > 1:
            # Multiple relation types combined
            max_strength = e.get('max_strength', e['relation_strength'])
            title = (
                f"Relations ({num_types} types): {relation_types}\n"
                f"Boosted Strength: {e['relation_strength']:.2f}\n"
                f"Max Individual Strength: {max_strength:.2f}\n"
                f"Total Evidence Count: {int(e['evidence_count'])}\n"
                f"Details:\n{e['evidence_sample']}"
            )
        else:
            # Single relation type
            title = (
                f"{relation_types}\n"
                f"Strength: {e['relation_strength']:.2f}\n"
                f"Evidence: {e['evidence_sample']}\n"
                f"Count: {int(e['evidence_count'])}"
            )
        net.add_edge(
            e["src_platform_uid"],
            e["related_platform_uid"],
            value=w,
            title=title,
        )

    # Make interactions nicer
    net.set_options(
        """
        var options = {
          "interaction": {
            "hover": true,
            "multiselect": true,
            "navigationButtons": true
          },
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 120}
          }
        }
        """
    )

    return net.generate_html()


# ----------------------------
# Streamlit app
# ----------------------------
st.set_page_config(page_title="Poker Bot Network Graph (Mock)", layout="wide")
st.title("Poker bot / collusion network graph (mock data)")
st.caption("Player-only graph: nodes = players, edges = relation types (device/IP/username/ADL/played_together).")

players_df, edges_df = generate_mock_data(seed=7)

with st.sidebar:
    st.header("Filters")
    case_id = st.selectbox("Group case", sorted(players_df["group_case_id"].unique()))

    # Get case-specific data for filters
    case_players_temp = players_df[players_df["group_case_id"] == case_id].copy()
    
    available_types = sorted(edges_df.loc[edges_df["group_case_id"] == case_id, "relation_type"].unique())
    selected_types = st.multiselect("Relation types", available_types, default=available_types)

    min_strength = st.slider("Min relation strength", 0.0, 1.0, 0.45, 0.05)
    
    st.divider()
    st.subheader("Player Filters")
    
    # Platform UID filter
    available_uids = sorted(case_players_temp["platform_uid"].unique())
    selected_uids = st.multiselect(
        "Platform UID (players)", 
        available_uids, 
        default=[],
        help="Select specific players to show. Leave empty to show all."
    )
    
    # Device filter
    available_devices = sorted(case_players_temp["primary_device"].unique())
    selected_devices = st.multiselect(
        "Primary Device", 
        available_devices, 
        default=[],
        help="Filter by device. Leave empty to show all devices."
    )
    
    # IP filter
    available_ips = sorted(case_players_temp["primary_ip"].unique())
    selected_ips = st.multiselect(
        "Primary IP", 
        available_ips, 
        default=[],
        help="Filter by IP address. Leave empty to show all IPs."
    )
    
    # Status filter
    available_statuses = sorted(case_players_temp["status"].unique())
    selected_statuses = st.multiselect(
        "Status", 
        available_statuses, 
        default=available_statuses,
        help="Filter by player status (active/watched/banned)."
    )
    
    # Bot score threshold
    min_bot_score = st.slider(
        "Min Bot Score ML", 
        0.0, 1.0, 0.0, 0.05,
        help="Minimum bot score to include players"
    )
    
    # ADL score threshold
    min_adl_score = st.slider(
        "Min ADL Score", 
        0, 100, 0, 5,
        help="Minimum ADL score to include players"
    )
    
    # Suspicious hand count threshold
    min_suspicious_hands = st.slider(
        "Min Suspicious Hands", 
        0, 20, 0, 1,
        help="Minimum suspicious hand count to include players"
    )
    
    # Edge count threshold
    min_edge_count = st.slider(
        "Min Edge Count (connections)", 
        0, 20, 0, 1,
        help="Minimum number of edges/connections a player must have. Use this to find highly connected players."
    )
    
    st.divider()
    st.subheader("Display Options")
    only_connected = st.checkbox("Show only connected players", value=True)
    only_suspicious_src = st.checkbox("Edges only from suspicious players", value=True)
    only_multi_relation_edges = st.checkbox(
        "Show only edges with multiple relation types", 
        value=False,
        help="Filter to show only edges where two players have multiple types of relationships (e.g., same device + played together)"
    )

    st.divider()
    st.subheader("Graph size controls")
    max_edges = st.slider("Max edges (after filtering)", 50, 2000, 400, 50)
    graph_height = st.slider("Graph height (px)", 450, 1200, 720, 50)

# Filter case
case_players = players_df[players_df["group_case_id"] == case_id].copy()
case_edges = edges_df[edges_df["group_case_id"] == case_id].copy()

# Apply player filters
if selected_uids:
    case_players = case_players[case_players["platform_uid"].isin(selected_uids)]

if selected_devices:
    case_players = case_players[case_players["primary_device"].isin(selected_devices)]

if selected_ips:
    case_players = case_players[case_players["primary_ip"].isin(selected_ips)]

if selected_statuses:
    case_players = case_players[case_players["status"].isin(selected_statuses)]

case_players = case_players[case_players["bot_score_ml"] >= min_bot_score]
case_players = case_players[case_players["adl_score"] >= min_adl_score]
case_players = case_players[case_players["suspicious_hand_count"] >= min_suspicious_hands]

# Apply edge filters
case_edges = case_edges[case_edges["relation_type"].isin(selected_types)]
case_edges = case_edges[case_edges["relation_strength"] >= min_strength]

# Calculate edge counts for each player (before filtering players)
# Count both outgoing (src) and incoming (related) edges
edge_counts = {}
all_player_uids = set(case_players["platform_uid"])
for uid in all_player_uids:
    outgoing = len(case_edges[case_edges["src_platform_uid"] == uid])
    incoming = len(case_edges[case_edges["related_platform_uid"] == uid])
    # Count unique connections (avoid double-counting bidirectional edges)
    unique_connections = len(set(
        list(case_edges[case_edges["src_platform_uid"] == uid]["related_platform_uid"]) +
        list(case_edges[case_edges["related_platform_uid"] == uid]["src_platform_uid"])
    ))
    edge_counts[uid] = unique_connections

# Add edge count to players dataframe
case_players["edge_count"] = case_players["platform_uid"].map(edge_counts).fillna(0).astype(int)

# Filter players by edge count
case_players = case_players[case_players["edge_count"] >= min_edge_count]

# Filter edges to only include connections between filtered players
filtered_player_set = set(case_players["platform_uid"])
case_edges = case_edges[
    (case_edges["src_platform_uid"].isin(filtered_player_set)) &
    (case_edges["related_platform_uid"].isin(filtered_player_set))
]

if only_suspicious_src:
    suspicious_set = set(case_players.loc[case_players["suspicious_hand_count"] > 0, "platform_uid"])
    case_edges = case_edges[case_edges["src_platform_uid"].isin(suspicious_set)]

# Aggregate multiple edges between same nodes into single edge with combined relation types
# PyVis doesn't support visualizing multiple parallel edges, so we combine them
if not case_edges.empty:
    aggregated_edges = []
    for (src, dst), group in case_edges.groupby(["src_platform_uid", "related_platform_uid"]):
        if len(group) > 1:
            # Multiple relation types - combine them
            relation_types = sorted(group["relation_type"].tolist())
            num_types = len(relation_types)
            
            # Better approach: Use evidence-weighted average strength, then boost for multiple types
            # This gives more weight to relations with more evidence (e.g., 100 hands vs 1 device match)
            total_evidence = group["evidence_count"].sum()
            if total_evidence > 0:
                weighted_strength = sum(group["relation_strength"] * group["evidence_count"]) / total_evidence
            else:
                weighted_strength = group["relation_strength"].mean()
            
            # Boost strength for multiple relation types (more suspicious when multiple signals align)
            # Formula: weighted_strength * (1 + 0.25 * (num_types - 1))
            # This means: 1 type = 1.0x, 2 types = 1.25x, 3 types = 1.5x, etc.
            boosted_strength = min(1.0, weighted_strength * (1 + 0.25 * (num_types - 1)))
            
            max_strength = group["relation_strength"].max()
            evidence_samples = " | ".join([f"{row['relation_type']}: {row['evidence_sample']}" 
                                          for _, row in group.iterrows()])
            
            aggregated_edges.append({
                "group_case_id": group.iloc[0]["group_case_id"],
                "src_platform_uid": src,
                "related_platform_uid": dst,
                "relation_type": ", ".join(relation_types),  # Combined types
                "relation_strength": float(boosted_strength),  # Use boosted strength for visualization
                "max_strength": float(max_strength),  # Keep original max for reference
                "num_relation_types": num_types,  # Track number of types
                "evidence_count": int(total_evidence),
                "evidence_sample": evidence_samples,
                "computed_ts": group.iloc[0]["computed_ts"],
            })
        else:
            # Single relation type - keep as is
            row = group.iloc[0].to_dict()
            row["num_relation_types"] = 1
            row["max_strength"] = row["relation_strength"]
            aggregated_edges.append(row)
    
    case_edges = pd.DataFrame(aggregated_edges)
    
    # Filter to show only edges with multiple relation types if requested
    if only_multi_relation_edges:
        case_edges = case_edges[case_edges["num_relation_types"] > 1].copy()

# Cap edges for usability
case_edges = case_edges.sort_values("relation_strength", ascending=False).head(max_edges)

# Optionally restrict nodes to connected ones
if only_connected and not case_edges.empty:
    connected = set(case_edges["src_platform_uid"]).union(set(case_edges["related_platform_uid"]))
    case_players = case_players[case_players["platform_uid"].isin(connected)].copy()

# Layout
left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.subheader("Network graph")
    st.write(
        f"Nodes: **{len(case_players)}**  |  "
        f"Edges: **{len(case_edges)}**  |  "
        f"Types: **{', '.join(selected_types) if selected_types else 'none'}**"
    )

    if len(case_players) == 0 or len(case_edges) == 0:
        st.warning("No nodes/edges match your filters. Lower the strength threshold or enable more relation types.")
    else:
        html = build_network_html(case_players, case_edges, height_px=graph_height)
        components.html(html, height=graph_height + 40, scrolling=True)

        st.download_button(
            "Download graph HTML",
            data=html.encode("utf-8"),
            file_name=f"network_{case_id.replace('__','_')}.html",
            mime="text/html",
        )

with right:
    st.subheader("Tables (mock)")
    st.markdown("**Players (nodes)**")
    st.dataframe(
        case_players.sort_values(["edge_count", "bot_score_ml"], ascending=[False, False])[
            [
                "platform_uid",
                "username",
                "edge_count",
                "status",
                "bot_score_ml",
                "adl_score",
                "suspicious_hand_count",
                "marked_hand_count",
                "primary_device",
                "primary_ip",
                "last_seen_ts",
            ]
        ],
        use_container_width=True,
        height=320,
    )

    st.download_button(
        "Download players CSV",
        data=case_players.to_csv(index=False).encode("utf-8"),
        file_name=f"players_{case_id.replace('__','_')}.csv",
        mime="text/csv",
    )

    st.markdown("**Relations (edges)**")
    # Show num_relation_types column if it exists
    display_cols = [
        "src_platform_uid",
        "related_platform_uid",
        "relation_type",
        "relation_strength",
        "evidence_count",
        "evidence_sample",
    ]
    if "num_relation_types" in case_edges.columns:
        display_cols.insert(3, "num_relation_types")  # Insert after relation_type
    st.dataframe(
        case_edges[display_cols],
        use_container_width=True,
        height=320,
    )

    st.download_button(
        "Download edges CSV",
        data=case_edges.to_csv(index=False).encode("utf-8"),
        file_name=f"edges_{case_id.replace('__','_')}.csv",
        mime="text/csv",
    )

st.divider()
st.markdown(
    """
### What to look for in the mock graph
- **Stars**: one suspicious player connected to many → “controller” / shared infra
- **Dense clusters**: many suspicious players interconnected → ring / farm
- **Bridging nodes**: player connecting two clusters → reseller / shared VPN / shared device rotation
"""
)
