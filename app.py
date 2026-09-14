import streamlit as st
import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import urllib.request
import matplotlib.font_manager as fm

# ---- 日本語フォント（IPAexゴシック）の自動設定 ----
FONT_PATH = "IPAexGothic.ttf"
if not os.path.exists(FONT_PATH):
    font_url = "https://github.com/google/fonts/raw/main/ofl/ipaexgothic/IPAexGothic.ttf"
    try:
        urllib.request.urlretrieve(font_url, FONT_PATH)
    except Exception:
        pass

if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = font_prop.get_name()

# ページの初期設定
st.set_page_config(page_title="ハエ生体脳 vs Q学習AI シミュレーター", layout="wide")
st.title("🧠 ハエ生体脳AI vs 🤖 普通のQ学習AI 比較シミュレーター")
st.write("「挑戦実行」を押すとAIがゲームに挑戦します。サイドバーからAIモデルを切り替えて比較できます。")

# 1. ハエの脳データ
@st.cache_resource
def load_brain():
    real_synapses = [
        (720575940616744816, 720575940621415232),
        (720575940621415232, 720575940613215280),
        (720575940613215280, 720575940630114000),
        (720575940630114000, 720575940625441111),
        (720575940625441111, 720575940619882222),
        (720575940619882222, 720575940641223333),
        (720575940641223333, 720575940611554444),
        (720575940611554444, 720575940622775555),
    ]
    G = nx.DiGraph()
    G.add_edges_from(real_synapses)
    return G

base_network = load_brain()
nodes_list = list(base_network.nodes())

# 2. 生体脳（ハエ）AIクラス
class FlyBrainAI:
    def __init__(self, network):
        self.network = network
        self.jump_threshold = random.uniform(5.0, 15.0)
        self.distance_weight = random.uniform(0.5, 1.5)

    def decide_action(self, bird_y, pipe_x, pipe_y):
        if bird_y > 15: return False, 0.0
        if bird_y < 5: return True, 1.0
        
        base_signal = random.random() * self.network.number_of_nodes()
        if pipe_x < 15:
            height_diff = (pipe_y + 3.5) - bird_y
            brain_signal = base_signal + (height_diff * self.distance_weight)
        else:
            brain_signal = base_signal
            
        is_jump = brain_signal > self.jump_threshold
        signal_intensity = min(1.0, max(0.0, brain_signal / 20.0))
        return is_jump, signal_intensity

    def mutate(self):
        child = FlyBrainAI(self.network)
        child.jump_threshold = max(2.0, self.jump_threshold + random.uniform(-1.0, 1.0))
        child.distance_weight = max(0.1, self.distance_weight + random.uniform(-0.2, 0.2))
        return child

# 3. 普通のQ学習AIクラス
class QLearningAI:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha     # 学習率
        self.gamma = gamma     # 割引率
        self.epsilon = epsilon # 探索率
        
    def get_state(self, bird_y, pipe_x, pipe_y):
        y_discrete = int(bird_y // 2)
        x_discrete = int(pipe_x // 3)
        diff_discrete = int(((pipe_y + 3.5) - bird_y) // 2)
        return (y_discrete, x_discrete, diff_discrete)

    def decide_action(self, bird_y, pipe_x, pipe_y):
        state = self.get_state(bird_y, pipe_x, pipe_y)
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
            
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            action = int(np.argmax(self.q_table[state]))
            
        is_jump = (action == 1)
        return is_jump, float(action)

    def update_q(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
            
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - predict)

# 4. セッション状態の初期化
if 'fly_gen' not in st.session_state:
    st.session_state.fly_gen = 0
    st.session_state.fly_scores = []
    st.session_state.best_fly_score = -1
    st.session_state.best_fly_brain = FlyBrainAI(base_network)
    st.session_state.current_fly_brain = st.session_state.best_fly_brain

if 'q_gen' not in st.session_state:
    st.session_state.q_gen = 0
    st.session_state.q_scores = []
    st.session_state.q_agent = QLearningAI()

# サイドバー設定
st.sidebar.header("⚙️ 実験設定")
ai_mode = st.sidebar.radio("AIモデルを選択", ["🧠 ハエ生体脳AI", "🤖 普通のQ学習AI"])

# 画像ファイルの探索・読み込み
def get_image_path(filename):
    candidates = [
        filename,
        os.path.join(os.path.expanduser("~"), "Desktop", filename),
        os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", filename)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

fly_img_path = get_image_path("fly.png")
robot_img_path = get_image_path("ai_character02.png")

# 5. 操作ボタン
col1, col2 = st.columns(2)
with col1:
    btn_next = st.button(f"挑戦実行 🚀 ({ai_mode})", use_container_width=True)
with col2:
    btn_reset = st.button("全データリセット 🔄", use_container_width=True)

if btn_reset:
    st.session_state.fly_gen = 0
    st.session_state.fly_scores = []
    st.session_state.best_fly_score = -1
    st.session_state.best_fly_brain = FlyBrainAI(base_network)
    st.session_state.current_fly_brain = st.session_state.best_fly_brain
    
    st.session_state.q_gen = 0
    st.session_state.q_scores = []
    st.session_state.q_agent = QLearningAI()
    st.success("🔄 すべての実験データをリセットしました！")

pos = nx.spring_layout(base_network, seed=42)

# 6. ゲーム実行・シミュレーション処理
if btn_next:
    is_fly_mode = ("ハエ生体脳AI" in ai_mode)
    if is_fly_mode:
        st.session_state.fly_gen += 1
        gen_label = f"ハエ No.{st.session_state.fly_gen}"
        active_img_path = fly_img_path
    else:
        st.session_state.q_gen += 1
        gen_label = f"Q学習 試行 {st.session_state.q_gen}"
        active_img_path = robot_img_path
        
    bird_y = 10.0
    pipe_x = 30.0
    pipe_gap = 7.0 
    pipe_y = float(random.randint(3, 10))
    score = 0
    game_over = False
    step_count = 0
    placeholder = st.empty()
    
    img = plt.imread(active_img_path) if active_img_path else None

    while not game_over:
        step_count += 1
        
        if is_fly_mode:
            is_jump, signal_intensity = st.session_state.current_fly_brain.decide_action(bird_y, pipe_x, pipe_y)
            action_idx = 0
        else:
            current_state = st.session_state.q_agent.get_state(bird_y, pipe_x, pipe_y)
            is_jump, _ = st.session_state.q_agent.decide_action(bird_y, pipe_x, pipe_y)
            action_idx = 1 if is_jump else 0

        # 移動処理
        if is_jump:
            bird_y += 1.4
        else:
            bird_y -= 0.9
        pipe_x -= 1.2
        
        passed_pipe = False
        if pipe_x < 0:
            pipe_x = 30.0
            pipe_y = float(random.randint(3, 10))
            score += 1 
            passed_pipe = True
            
        if bird_y < 0 or bird_y > 20: 
            game_over = True
        if 1 <= pipe_x <= 4 and (bird_y < pipe_y or bird_y > pipe_y + pipe_gap): 
            game_over = True

        # Q学習のQ値更新
        if not is_fly_mode:
            next_state = st.session_state.q_agent.get_state(bird_y, pipe_x, pipe_y)
            reward = -100.0 if game_over else (15.0 if passed_pipe else 0.1)
            st.session_state.q_agent.update_q(current_state, action_idx, reward, next_state)

        # 2ステップに1回描画（高速化）
        if step_count % 2 == 0 or game_over:
            fig, (ax_game, ax_graph, ax_brain) = plt.subplots(1, 3, figsize=(15, 4))
            
            # 1. ゲーム画面
            ax_game.set_xlim(0, 30)
            ax_game.set_ylim(0, 20)
            if img is not None:
                ax_game.imshow(img, extent=[1.5, 4.5, bird_y - 1.5, bird_y + 1.5], zorder=3)
            else:
                ax_game.text(3, bird_y, "🪰" if is_fly_mode else "🤖", fontsize=24, ha='center', va='center') 
                
            ax_game.bar(pipe_x, pipe_y, width=2, color='green')
            ax_game.bar(pipe_x, 20 - (pipe_y + pipe_gap), width=2, bottom=pipe_y + pipe_gap, color='green')
            ax_game.set_title(f"{gen_label} | スコア: {score}")
            ax_game.set_xticks([]); ax_game.set_yticks([])
            
            # 2. スコア学習曲線グラフ
            if st.session_state.fly_scores:
                ax_graph.plot(range(1, len(st.session_state.fly_scores) + 1), st.session_state.fly_scores, 
                              marker='o', color='deeppink', label='ハエ生体脳AI', linewidth=2)
            if st.session_state.q_scores:
                ax_graph.plot(range(1, len(st.session_state.q_scores) + 1), st.session_state.q_scores, 
                              marker='s', color='limegreen', label='普通のQ学習AI', linewidth=2)
                
            max_len = max(10, len(st.session_state.fly_scores), len(st.session_state.q_scores))
            ax_graph.set_xlim(0.5, max_len + 0.5)
            ax_graph.set_title("AI学習パフォーマンス比較")
            ax_graph.set_xlabel("試行回数 / 世代")
            ax_graph.set_ylabel("スコア")
            if st.session_state.fly_scores or st.session_state.q_scores:
                ax_graph.legend(loc='upper left')
            ax_graph.grid(True)
            
            # 3. 脳神経回路 / Q学習状態表示
            if is_fly_mode:
                active_idx = step_count % len(nodes_list)
                node_colors = []
                node_sizes = []
                for i in range(len(nodes_list)):
                    if i == active_idx:
                        node_colors.append('#FF1493' if is_jump else '#00FFFF')
                        node_sizes.append(280)
                    else:
                        node_colors.append('#D3D3D3')
                        node_sizes.append(150)
                        
                nx.draw_networkx_nodes(base_network, pos, ax=ax_brain, node_color=node_colors, node_size=node
