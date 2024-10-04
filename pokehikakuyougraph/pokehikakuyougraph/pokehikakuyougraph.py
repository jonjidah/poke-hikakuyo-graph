import numpy as np
import itertools
import streamlit as st
import plotly.graph_objs as go
import pandas as pd

# ポケモンの単一タイプを定義
single_types = [
    'Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting',
    'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost',
    'Dragon', 'Dark', 'Steel', 'Fairy'
]

# タイプ相性チャートを定義
type_chart = {
    'Normal':     {'Rock': 0.5, 'Ghost': 0, 'Steel': 0.5},
    'Fire':       {'Fire': 0.5, 'Water': 0.5, 'Grass': 2, 'Ice': 2, 'Bug': 2, 'Rock': 0.5, 'Dragon': 0.5, 'Steel': 2},
    'Water':      {'Fire': 2, 'Water': 0.5, 'Grass': 0.5, 'Ground': 2, 'Rock': 2, 'Dragon': 0.5},
    'Electric':   {'Water': 2, 'Electric': 0.5, 'Grass': 0.5, 'Ground': 0, 'Flying': 2, 'Dragon': 0.5},
    'Grass':      {'Fire': 0.5, 'Water': 2, 'Grass': 0.5, 'Poison': 0.5, 'Ground': 2, 'Flying': 0.5, 'Bug': 0.5, 'Rock': 2, 'Dragon': 0.5, 'Steel': 0.5},
    'Ice':        {'Fire': 0.5, 'Water': 0.5, 'Grass': 2, 'Ice': 0.5, 'Ground': 2, 'Flying': 2, 'Dragon': 2, 'Steel': 0.5},
    'Fighting':   {'Normal': 2, 'Ice': 2, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 0.5, 'Bug': 0.5, 'Rock': 2, 'Ghost': 0, 'Dark': 2, 'Steel': 2, 'Fairy': 0.5},
    'Poison':     {'Grass': 2, 'Poison': 0.5, 'Ground': 0.5, 'Rock': 0.5, 'Ghost': 0.5, 'Steel': 0, 'Fairy': 2},
    'Ground':     {'Fire': 2, 'Electric': 2, 'Grass': 0.5, 'Poison': 2, 'Flying': 0, 'Bug': 0.5, 'Rock': 2, 'Steel': 2},
    'Flying':     {'Electric': 0.5, 'Grass': 2, 'Fighting': 2, 'Bug': 2, 'Rock': 0.5, 'Steel': 0.5},
    'Psychic':    {'Fighting': 2, 'Poison': 2, 'Psychic': 0.5, 'Dark': 0, 'Steel': 0.5},
    'Bug':        {'Fire': 0.5, 'Grass': 2, 'Fighting': 0.5, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 2, 'Ghost': 0.5, 'Dark': 2, 'Steel': 0.5, 'Fairy': 0.5},
    'Rock':       {'Fire': 2, 'Ice': 2, 'Fighting': 0.5, 'Ground': 0.5, 'Flying': 2, 'Bug': 2, 'Steel': 0.5},
    'Ghost':      {'Normal': 0, 'Psychic': 2, 'Ghost': 2, 'Dark': 0.5},
    'Dragon':     {'Dragon': 2, 'Steel': 0.5, 'Fairy': 0},
    'Dark':       {'Fighting': 0.5, 'Psychic': 2, 'Ghost': 2, 'Dark': 0.5, 'Fairy': 0.5},
    'Steel':      {'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Ice': 2, 'Rock': 2, 'Steel': 0.5, 'Fairy': 2},
    'Fairy':      {'Fire': 0.5, 'Fighting': 2, 'Poison': 0.5, 'Dragon': 2, 'Dark': 2, 'Steel': 0.5},
}

def get_effectiveness(attacking_type, defending_types):
    """
    攻撃タイプが防御タイプまたはデュアルタイプに対して持つ効果倍率を計算します。
    デュアルタイプの場合、各防御タイプに対する倍率を掛け合わせます。
    """
    multiplier = 1.0
    for dtype in defending_types:
        if dtype in type_chart.get(attacking_type, {}):
            multiplier *= type_chart[attacking_type][dtype]
        else:
            multiplier *= 1.0
    return multiplier

# デュアルタイプ（組み合わせ）を生成
dual_type_pairs = list(itertools.combinations(single_types, 2))

# 単一タイプとデュアルタイプを結合
all_types = single_types + [f"{t1}/{t2}" for t1, t2 in dual_type_pairs]

# 全タイプ数
num_types = len(all_types)  # 18 + 153 = 171

# タイプ名とインデックスのマッピングを作成
type_to_index = {type_name: idx for idx, type_name in enumerate(all_types)}
index_to_type = {idx: type_name for type_name, idx in type_to_index.items()}

# 効果倍率行列を初期化（171 x 171）
# 行: 攻撃タイプ
# 列: 防御タイプ
# 値: 効果倍率
effectiveness_matrix = np.ones((num_types, num_types))

for atk_idx, atk_type in enumerate(all_types):
    # デュアルタイプを分割
    atk_components = atk_type.split('/')
    for def_idx, def_type in enumerate(all_types):
        def_components = def_type.split('/')
        if len(atk_components) == 1:
            # 単一タイプの攻撃
            eff = get_effectiveness(atk_components[0], def_components)
        else:
            # デュアルタイプの攻撃: 2つの単一タイプのうち最大の効果倍率を使用
            eff1 = get_effectiveness(atk_components[0], def_components)
            eff2 = get_effectiveness(atk_components[1], def_components)
            eff = max(eff1, eff2)
        effectiveness_matrix[atk_idx, def_idx] = eff

# Streamlitアプリケーションの開始
st.title("ポケモンタイプランキングの可視化（攻撃力と防御力のバランス）")

st.markdown("""
このアプリケーションでは、ポケモンの各タイプに対して**攻撃力**と**防御力**のバランスを取ったスコアリングシステムを導入し、`defense_weight`を0から1まで0.1刻みで変更しながら、各タイプの総合スコアの変動を確認できます。

**操作方法:**
1. **Defense Weight**をスライダーで調整して、攻撃力と防御力のバランスを変更します。
2. **表示するタイプを2つのグループに分けて選択**し、それぞれ異なる色（赤と青）でグラフに表示します。グループ内では、選択順に色が薄くなります。
3. グラフ上でマウスをホバーすると、特定のタイプの詳細情報が表示されます。
4. **詳細情報**として、選択したタイプの最大スコアとそのスコアを得た`defense_weight`を確認できます。
""")

# ユーザー入力: グループ1とグループ2の選択
st.sidebar.header("タイプのグループ分け")

group1_types = st.sidebar.multiselect(
    "グループ1（赤色）に含めるポケモンタイプを選択してください",
    options=all_types,
    default=[]
)

group2_types = st.sidebar.multiselect(
    "グループ2（青色）に含めるポケモンタイプを選択してください",
    options=[t for t in all_types if t not in group1_types],
    default=[]
)

# 残りのタイプは表示しない
selected_types = group1_types + group2_types

if not selected_types:
    st.warning("表示するポケモンタイプをグループ1またはグループ2から選択してください。")
    st.stop()

# シミュレーションのパラメータ
defense_weights = np.arange(0.0, 1.1, 0.1)  # 0.0から1.0まで0.1刻み

# 各タイプのスコアを保存する辞書
type_score_history = {type_name: [] for type_name in selected_types}

# 各タイプの最大スコアと対応するdefense_weightを保存する辞書
type_max_info = {type_name: {'max_score': -np.inf, 'defense_weight': None} for type_name in selected_types}

# シミュレーションの実行
for defense_weight in defense_weights:
    attack_weight = 1.0 - defense_weight

    # タイプ分布を均等に初期化
    type_distribution = np.ones(num_types) / num_types

    # 攻撃力スコアと防御力スコアを保存するための配列を初期化
    attack_scores = np.zeros(num_types)
    defense_scores = np.zeros(num_types)

    # シミュレーションのパラメータ
    iterations = 10000
    learning_rate = 0.1

    # 防御の重み付け係数（1より大きい値で防御を強調）
    defense_weight_factor = 2.0

    # シミュレーションの実行
    for _ in range(iterations):
        # 攻撃効果の計算: 各タイプの攻撃が現在の分布に対してどれだけ効果的か
        attack_effectiveness = effectiveness_matrix.dot(type_distribution)

        # 防御効果の計算:
        # 各タイプが受ける総ダメージを計算し、その逆数を取ることで防御力を評価
        damage_received = effectiveness_matrix.T.dot(type_distribution)
        defense_effectiveness = 1 / (damage_received + 1e-6)  # ゼロ除算を避けるために小さな値を追加

        # 防御の重み付けを適用
        defense_effectiveness *= defense_weight_factor

        # 攻撃スコアと防御スコアを保存
        attack_scores += attack_effectiveness
        defense_scores += defense_effectiveness

        # 新しいタイプ分布の計算
        new_distribution = attack_effectiveness + defense_effectiveness

        # 正規化
        new_distribution /= new_distribution.sum()

        # 分布の更新（学習率を適用して安定性を確保）
        type_distribution = (1 - learning_rate) * type_distribution + learning_rate * new_distribution

    # 攻撃力スコアと防御力スコアを正規化
    attack_scores_normalized = attack_scores / np.linalg.norm(attack_scores, ord=1)
    defense_scores_normalized = defense_scores / np.linalg.norm(defense_scores, ord=1)

    # 総合スコアの計算
    total_scores = attack_weight * attack_scores_normalized + defense_weight * defense_scores_normalized

    # 総合スコアを正規化
    total_scores /= total_scores.sum()

    # タイプごとのスコアをリストにまとめる
    type_scores = list(zip(all_types, total_scores))

    # スコアに基づいて降順にソート
    type_scores.sort(key=lambda x: x[1], reverse=True)

    # ランキングを割り当て
    ranked_types = [(rank + 1, type_name, score) for rank, (type_name, score) in enumerate(type_scores)]

    # 各タイプのスコア履歴を更新
    for rank, type_name, score in ranked_types:
        if type_name in selected_types:
            type_score_history[type_name].append(score)

            # 各タイプの最大スコアと対応するdefense_weightを更新
            if score > type_max_info[type_name]['max_score']:
                type_max_info[type_name]['max_score'] = score
                type_max_info[type_name]['defense_weight'] = defense_weight

# 全体のスコア履歴からグラフのy軸範囲を固定
all_scores = []
for scores in type_score_history.values():
    all_scores.extend(scores)
if all_scores:
    max_score = max(all_scores)
    y_axis_max = max_score * 1.1  # 10%余裕を持たせる
    y_axis_min = 0
else:
    y_axis_max = 1
    y_axis_min = 0

# Plotlyのグラフ作成
fig = go.Figure()

# 色の設定: グループ1は赤、グループ2は青
def get_shaded_color(base_color, index, total):
    """
    base_color: 'red' or 'blue'
    index: 0-based index
    total: total number of lines in the group
    """
    if base_color == 'red':
        rgb = (255, 0, 0)
    elif base_color == 'blue':
        rgb = (0, 0, 255)
    else:
        rgb = (0, 0, 0)
    
    # 薄くなるようにアルファを調整（1.0から0.3まで）
    alpha = 1.0 - (index / total) * 0.7
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"

# グループごとに色を設定して線を追加
def add_traces(fig, group_types, color):
    total = len(group_types)
    for idx, type_name in enumerate(group_types):
        color_shade = get_shaded_color(color, idx, total)
        scores = type_score_history[type_name]
        max_info = type_max_info[type_name]
        fig.add_trace(go.Scatter(
            x=defense_weights,
            y=scores,
            mode='lines',
            name=type_name,
            line=dict(color=color_shade, width=2),
            hovertemplate=(
                f"Type: {type_name}<br>"
                "Defense Weight: %{x}<br>"
                "Score: %{y:.6f}<br>"
                f"Max Score: {max_info['max_score']:.6f}<br>"
                f"Defense Weight at Max: {max_info['defense_weight']:.1f}<extra></extra>"
            )
        ))

# グループ1とグループ2のトレースを追加
add_traces(fig, group1_types, 'red')
add_traces(fig, group2_types, 'blue')

# グラフのレイアウト設定
fig.update_layout(
    title='ポケモンタイプの総合スコアの変動（攻撃力と防御力のバランス）',
    xaxis_title='Defense Weight',
    yaxis_title='Total Score',
    xaxis=dict(range=[0, 1], dtick=0.1),
    yaxis=dict(range=[y_axis_min, y_axis_max]),
    hovermode='closest'
)

# StreamlitでPlotlyグラフを表示
st.plotly_chart(fig, use_container_width=True)

# インタラクティブな詳細情報の表示
st.markdown("### タイプの詳細情報")

# ユーザーが選択したタイプの詳細を表示するためのインターフェース
selected_type_detail = st.selectbox("詳細を表示するタイプを選択してください", options=selected_types)

if selected_type_detail:
    max_defense_weight = type_max_info[selected_type_detail]['defense_weight']
    max_score = type_max_info[selected_type_detail]['max_score']
    st.write(f"**タイプ:** {selected_type_detail}")
    st.write(f"**最大スコア:** {max_score:.6f}")
    st.write(f"**最大スコアを得たDefense Weight:** {max_defense_weight:.1f}")

# まとめとしてランキング表を表示
st.markdown("### 最終ポケモンタイプランキング（攻撃力と防御力のバランス）")

# 最終defense_weight = 1.0の場合のランキングを表示
final_defense_weight = 1.0
final_attack_weight = 1.0 - final_defense_weight

# 最終defense_weightにおける総合スコアを取得
final_total_scores = {type_name: type_score_history[type_name][-1] for type_name in selected_types}

# スコアに基づいて降順にソート
final_type_scores = sorted(final_total_scores.items(), key=lambda x: x[1], reverse=True)

# ランキングを割り当て
final_ranked_types = [(rank + 1, type_name, score) for rank, (type_name, score) in enumerate(final_type_scores)]

# 表を作成
df_final = pd.DataFrame(final_ranked_types, columns=['Rank', 'Type', 'Score'])
st.dataframe(df_final)

# 上位20タイプと下位20タイプの表示
st.markdown("### 上位20ポケモンタイプ（攻撃力と防御力のバランス）")

top_20 = final_ranked_types[:20]
df_top_20 = pd.DataFrame(top_20, columns=['Rank', 'Type', 'Score'])
st.dataframe(df_top_20)

st.markdown("### 下位20ポケモンタイプ（攻撃力と防御力のバランス）")

bottom_20 = final_ranked_types[-20:]
df_bottom_20 = pd.DataFrame(bottom_20, columns=['Rank', 'Type', 'Score'])
st.dataframe(df_bottom_20)
