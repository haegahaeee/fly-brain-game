import streamlit as st
import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import os
import numpy as np

# ページの初期設定
st.set_page_config(page_title="Fly Brain vs Q-Learning Simulator", layout="wide")
st.title("🧠 生体脳AI vs 🤖 Q学習AI 比較シミュレーター")
st.write("「挑戦実行」を押すとAIがゲームに挑戦します。サイドバーからモデルを切り替えて比較できます。")

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
st.sidebar.header("⚙️ Experiment Setup")
ai_mode = st.sidebar.radio("Select AI Model", ["Fly Brain (Bio-AI)", "Standard Q-Learning AI"])

# 画像パス探索
def get_fly_image_path():
    candidates = [
        "fly.png",
        os.path.join(os.path.expanduser("~"), "Desktop", "fly.png"),
        os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "fly.png")
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

img_path = get_fly_image_path()

# 5. 操作ボタン
col1, col2 = st.columns(2)
with col1:
    btn_next = st.button(f"Run Trial 🚀 ({ai_mode})", use_container_width=True)
with col2:
    btn_reset = st.button("Reset All Data 🔄", use_container_width=True)

if btn_reset:
    st.session_state.fly_gen = 0
    st.session_state.fly_scores = []
    st.session_state.best_fly_score = -1
    st.session_state.best_fly_brain = FlyBrainAI(base_network)
    st.session_state.current_fly_brain = st.session_state.best_fly_brain
    
    st.session_state.q_gen = 0
    st.session_state.q_scores = []
    st.session_state.q_agent = QLearningAI()
    st.success("🔄 All experiment data has been reset!")

pos = nx.spring_layout(base_network, seed=42)

# 6. ゲーム実行・シミュレーション処理
if btn_next:
    is_fly_mode = ("Fly Brain" in ai_mode)
    if is_fly_mode:
        st.session_state.fly_gen += 1
        gen_label = f"Fly Gen {st.session_state.fly_gen}"
    else:
        st.session_state.q_gen += 1
        gen_label = f"Q-Learning Gen {st.session_state.q_gen}"
        
    bird_y = 10.0
    pipe_x = 30.0
    pipe_gap = 7.0 
    pipe_y = float(random.randint(3, 10))
    score = 0
    game_over = False
    step_count = 0
    placeholder = st.empty()
    
    img = plt.imread(img_path) if img_path else None
    fig, (ax_game, ax_graph, ax_brain) = plt.subplots(1, 3, figsize=(15, 4))

    while not game_over:
        step_count += 1
        
        if is_fly_mode:
            is_jump, signal_intensity = st.session_state.current_fly_brain.decide_action(bird_y, pipe_x, pipe_y)
        else:
            current_state = st.session_state.q_agent.get_state(bird_y, pipe_x, pipe_y)
            is_jump, _ = st.session_state.q_agent.decide_action(bird_y, pipe_x, pipe_y)
            action_idx = 1 if is_jump else 0

        # 移動設定
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

        # Q値の更新
        if not is_fly_mode:
            next_state = st.session_state.q_agent.get_state(bird_y, pipe_x, pipe_y)
            if game_over:
                reward = -100.0
            elif passed_pipe:
                reward = 15.0
            else:
                reward = 0.1
            st.session_state.q_agent.update_q(current_state, action_idx, reward, next_state)

        # 描画高速化（2ステップに1回描画）
        if step_count % 2 == 0 or game_over:
            ax_game.clear()
            ax_graph.clear()
            ax_brain.clear()
            
            # 1. プレイ画面
            ax_game.set_xlim(0, 30)
            ax_game.set_ylim(0, 20)
            if img is not None:
                ax_game.imshow(img, extent=[1.5, 4.5, bird_y - 1.5, bird_y + 1.5], zorder=3)
            else:
                ax_game.text(3, bird_y, "🪰" if is_fly_mode else "🤖", fontsize=24, ha='center', va='center') 
                
            ax_game.bar(pipe_x, pipe_y, width=2, color='green')
            ax_game.bar(pipe_x, 20 - (pipe_y + pipe_gap), width=2, bottom=pipe_y + pipe_gap, color='green')
            ax_game.set_title(f"{gen_label} | Score: {score}")
            ax_game.set_xticks([]); ax_game.set_yticks([])
            
            # 2. 学習曲線比較グラフ
            if st.session_state.fly_scores:
                ax_graph.plot(range(1, len(st.session_state.fly_scores) + 1), st.session_state.fly_scores, 
                              marker='o', color='deeppink', label='Fly Brain (Bio-AI)', linewidth=2)
            if st.session_state.q_scores:
                ax_graph.plot(range(1, len(st.session_state.q_scores) + 1), st.session_state.q_scores, 
                              marker='s', color='limegreen', label='Standard Q-Learning', linewidth=2)
                
            max_len = max(10, len(st.session_state.fly_scores), len(st.session_state.q_scores))
            ax_graph.set_xlim(0.5, max_len + 0.5)
            ax_graph.set_title("AI Performance Comparison")
            ax_graph.set_xlabel("Trials / Generations")
            ax_graph.set_ylabel("Score")
            ax_graph.legend(loc='upper left')
            ax_graph.grid(True)
            
            # 3. 脳回路 / Qテーブル状態の描画
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
                        
                nx.draw_networkx_nodes(base_network, pos, ax=ax_brain, node_color=node_colors, node_size=node_sizes)
                nx.draw_networkx_edges(base_network, pos, ax=ax_brain, edge_color='#808080', arrows=True, arrowstyle='->', arrowsize=12, width=2)
                ax_brain.set_title("Fly Connectome Network")
                ax_brain.text(0, -1.2, "Action: JUMP!" if is_jump else "State: Cruising...", fontsize=13, fontweight='bold', color="crimson" if is_jump else "gray", ha='center')
            else:
                ax_brain.set_title("Q-Learning Status")
                states_count = len(st.session_state.q_agent.q_table)
                ax_brain.text(0.5, 0.6, f"Learned States: {states_count}", fontsize=14, ha='center')
                ax_brain.text(0.5, 0.4, f"Action: {'JUMP' if is_jump else 'STAY'}", fontsize=14, fontweight='bold', color='limegreen' if is_jump else 'blue', ha='center')
            
            ax_brain.axis('off')
            
            with placeholder.container():
                st.pyplot(fig)
            
    plt.close(fig)
        
    # 学習データの保存
    if is_fly_mode:
        st.session_state.fly_scores.append(score)
        if score >= st.session_state.best_fly_score:
            st.session_state.best_fly_score = score
            st.session_state.best_fly_brain = st.session_state.current_fly_brain
        st.session_state.current_fly_brain = st.session_state.best_fly_brain.mutate()
    else:
        st.session_state.q_scores.append(score)
        
    st.rerun()

# 7. 待機画面
if not btn_next:
    fig, (ax_game, ax_graph, ax_brain) = plt.subplots(1, 3, figsize=(15, 4))
    
    ax_game.text(15, 10, "Ready to Test", fontsize=18, color='gray', ha='center', va='center')
    ax_game.set_xlim(0, 30); ax_game.set_ylim(0, 20)
    ax_game.set_xticks([]); ax_game.set_yticks([])
    
    if st.session_state.fly_scores:
        ax_graph.plot(range(1, len(st.session_state.fly_scores) + 1), st.session_state.fly_scores, 
                      marker='o', color='deeppink', label='Fly Brain (Bio-AI)', linewidth=2)
    if st.session_state.q_scores:
        ax_graph.plot(range(1, len(st.session_state.q_scores) + 1), st.session_state.q_scores, 
                      marker='s', color='limegreen', label='Standard Q-Learning', linewidth=2)
        
    ax_graph.set_title("AI Performance Comparison")
    ax_graph.set_xlabel("Trials / Generations")
    ax_graph.set_ylabel("Score")
    if st.session_state.fly_scores or st.session_state.q_scores:
        ax_graph.legend(loc='upper left')
    ax_graph.grid(True)
    
    nx.draw_networkx_nodes(base_network, pos, ax=ax_brain, node_color='gray', node_size=150)
    nx.draw_networkx_edges(base_network, pos, ax=ax_brain, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=10)
    ax_brain.set_title("Fly Connectome Network")
    ax_brain.axis('off')
    
    st.pyplot(fig)
    plt.close(fig)
