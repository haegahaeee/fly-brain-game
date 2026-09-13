import streamlit as st
import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import os

# ページの初期設定
st.set_page_config(page_title="Fly Brain AI Simulator", layout="wide")
st.title("🧠 ハエの本物脳データ × Flappy Bird シミュレーター")
st.write("「転生」を押すとハエの脳がゲームに挑戦し、少しずつ進化していきます。")

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

# 2. ハエの脳AIクラス
class FlyBrainAI:
    def __init__(self, network):
        self.network = network
        self.jump_threshold = random.uniform(5.0, 15.0)
        self.distance_weight = random.uniform(0.5, 1.5)

    def decide_action(self, bird_y, pipe_x, pipe_y):
        if bird_y > 15: return False
        if bird_y < 5: return True
        base_signal = random.random() * self.network.number_of_nodes()
        if pipe_x < 15:
            height_diff = (pipe_y + 3.5) - bird_y
            brain_signal = base_signal + (height_diff * self.distance_weight)
        else:
            brain_signal = base_signal
        return brain_signal > self.jump_threshold

    def mutate(self):
        child = FlyBrainAI(self.network)
        child.jump_threshold = max(2.0, self.jump_threshold + random.uniform(-1.0, 1.0))
        child.distance_weight = max(0.1, self.distance_weight + random.uniform(-0.2, 0.2))
        return child

# 3. データ保存
if 'generation' not in st.session_state:
    st.session_state.generation = 0
    st.session_state.history_scores = []
    st.session_state.current_brain = FlyBrainAI(base_network)

# 4. 操作ボタン
col1, col2 = st.columns(2)
with col1:
    btn_next = st.button("転生 🚀", use_container_width=True)
with col2:
    btn_reset = st.button("リセット 🔄", use_container_width=True)

if btn_reset:
    st.session_state.generation = 0
    st.session_state.history_scores = []
    st.session_state.current_brain = FlyBrainAI(base_network)
    st.success("🔄 脳の進化と記録をリセットしました！1回目から再開できます。")

# 5. ゲーム開始
if btn_next:
    st.session_state.generation += 1
    bird_y = 10
    pipe_x = 30
    pipe_gap = 7 
    pipe_y = random.randint(3, 10)
    score = 0
    game_over = False
    placeholder = st.empty()
    
    # ユーザーのデスクトップにある「fly.png」の絶対パスを自動作成
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "fly.png")
    # OneDriveのデスクトップを使っている場合用の予備パス
    onedrive_path = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "fly.png")
    
    while not game_over:
        if st.session_state.current_brain.decide_action(bird_y, pipe_x, pipe_y):
            bird_y += 1.4  
        else:
            bird_y -= 1.0  
        pipe_x -= 1.0 
        
        if pipe_x < 0:
            pipe_x = 30
            pipe_y = random.randint(3, 10)
            score += 1 
            
        if bird_y < 0 or bird_y > 20: game_over = True
        if 1 <= int(pipe_x) <= 3 and (bird_y < pipe_y or bird_y > pipe_y + pipe_gap): game_over = True
        
        # --- 🎬 画面描画 ---
        fig, (ax_game, ax_graph) = plt.subplots(1, 2, figsize=(11, 4))
        
        ax_game.set_xlim(0, 30)
        ax_game.set_ylim(0, 20)
        
        # デスクトップの指定場所に画像があるかチェック
        target_path = None
        if os.path.exists(desktop_path):
            target_path = desktop_path
        elif os.path.exists(onedrive_path):
            target_path = onnedrive_path
            
        if target_path:
            img = plt.imread(target_path)
            ax_game.imshow(img, extent=[1.5, 4.5, bird_y - 1.5, bird_y + 1.5], zorder=3)
        else:
            ax_game.text(3, bird_y, "O", fontsize=18, ha='center', va='center', color='red') 
            
        ax_game.bar(pipe_x, pipe_y, width=2, color='green')
        ax_game.bar(pipe_x, 20 - (pipe_y + pipe_gap), width=2, bottom=pipe_y + pipe_gap, color='green')
        ax_game.set_title(f"Fly No.{st.session_state.generation} | Score: {score}")
        ax_game.set_xticks([]); ax_game.set_yticks([])
        
        temp_history = st.session_state.history_scores + [score]
        ax_graph.plot(range(1, len(temp_history) + 1), temp_history, marker='o', color='dodgerblue', linewidth=2)
        ax_graph.set_xlim(0.5, max(10, len(temp_history)) + 0.5)
        ax_graph.set_ylim(-0.5, max(temp_history) + 3)
        ax_graph.set_title("Fly Brain AI Learning Progress")
        ax_graph.set_xlabel("Gen (Fly No.)")
        ax_graph.set_ylabel("Score")
        ax_graph.grid(True)
        
        with placeholder.container():
            st.pyplot(fig)
        plt.close(fig)
        time.sleep(0.02)
        
    st.session_state.history_scores.append(score)
    if score >= max(st.session_state.history_scores) and score > 0:
        st.session_state.current_brain = st.session_state.current_brain.mutate()
    else:
        st.session_state.current_brain = FlyBrainAI(base_network)
    st.rerun()

if st.session_state.generation > 0 and not btn_next:
    fig, (ax_game, ax_graph) = plt.subplots(1, 2, figsize=(11, 4))
    ax_game.text(15, 10, "Game Over", fontsize=20, color='red', ha='center')
    ax_game.set_xticks([]); ax_game.set_yticks([])
    
    ax_graph.plot(range(1, len(st.session_state.history_scores) + 1), st.session_state.history_scores, marker='o', color='dodgerblue', linewidth=2)
    ax_graph.set_xlim(0.5, max(10, len(st.session_state.history_scores)) + 0.5)
    ax_graph.set_ylim(-0.5, max(st.session_state.history_scores) + 3)
    ax_graph.set_title("Fly Brain AI Learning Progress")
    ax_graph.grid(True)
    st.pyplot(fig)
    plt.close(fig)
