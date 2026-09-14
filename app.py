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
st.set_page_config(page_title="ハエの本物脳データ × Flappy Bird シミュレーター", layout="wide")
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
nodes_list = list(base_network.nodes())

# 2. ハエAIクラス
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

# 3. セッション状態の初期化
if 'generation' not in st.session_state:
    st.session_state.generation = 0
    st.session_state.scores = []
    st.session_state.best_score = -1
    st.session_state.best_brain = FlyBrainAI(base_network)
    st.session_state.current_brain = st.session_state.best_brain

# 画像の読み込み処理
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

# ボタン配置
col1, col2 = st.columns(2)
with col1:
    btn_next = st.button("転生 🚀", use_container_width=True)
with col2:
    btn_reset = st.button("リセット 🔄", use_container_width=True)

if btn_reset:
    st.session_state.generation = 0
    st.session_state.scores = []
    st.session_state.best_score = -1
    st.session_state.best_brain = FlyBrainAI(base_network)
    st.session_state.current_brain = st.session_state.best_brain
    st.success("🔄 実験データをリセットしました！")

pos = nx.spring_layout(base_network, seed=42)

# シミュレーション実行
if btn_next:
    st.session_state.generation += 1
    bird_y = 10.0
    pipe_x = 30.0
    pipe_gap = 7.0
    pipe_y = float(random.randint(3, 10))
    score = 0
    game_over = False
    step_count = 0
    placeholder = st.empty()
    
    img = plt.imread(fly_img_path) if fly_img_path else None

    while not game_over:
        step_count += 1
        is_jump, signal_intensity = st.session_state.current_brain.decide_action(bird_y, pipe_x, pipe_y)

        if is_jump:
            bird_y += 1.4
        else:
            bird_y -= 0.9
        pipe_x -= 1.2
        
        if pipe_x < 0:
            pipe_x = 30.0
            pipe_y = float(random.randint(3, 10))
            score += 1
            
        if bird_y < 0 or bird_y > 20:
            game_over = True
        if 1 <= pipe_x <= 4 and (bird_y < pipe_y or bird_y > pipe_y + pipe_gap):
            game_over = True

        if step_count % 2 == 0 or game_over:
            fig, (ax_game, ax_graph, ax_brain) = plt.subplots(1, 3, figsize=(15, 4))
            
            # ゲーム画面
            ax_game.set_xlim(0, 30)
            ax_game.set_ylim(0, 20)
            if img is not None:
                ax_game.imshow(img, extent=[1.5, 4.5, bird_y - 1.5, bird_y + 1.5], zorder=3)
            else:
                ax_game.text(3, bird_y, "🪰", fontsize=24, ha='center', va='center')
                
            ax_game.bar(pipe_x, pipe_y, width=2, color='green')
            ax_game.bar(pipe_x, 20 - (pipe_y + pipe_gap), width=2, bottom=pipe_y + pipe_gap, color='green')
            ax_game.set_title(f"Fly No.{st.session_state.generation} | Score: {score}")
            ax_game.set_xticks([]); ax_game.set_yticks([])
            
            # 学習経過グラフ
            if st.session_state.scores:
                ax_graph.plot(range(1, len(st.session_state.scores) + 1), st.session_state.scores, 
                              marker='o', color='dodgerblue', linewidth=2)
            max_len = max(10, len(st.session_state.scores))
            ax_graph.set_xlim(0.5, max_len + 0.5)
            ax_graph.set_title("Fly Brain AI Learning Progress")
            ax_graph.set_xlabel("Gen (Fly No.)")
            ax_graph.set_ylabel("Score")
            ax_graph.grid(True)
            
            # コネクトーム（脳回路）
            active_idx = step_count % len(nodes_list)
            node_colors = []
            node_sizes = []
            for i in range(len(nodes_list)):
                if i == active_idx:
                    node_colors.append('#FF149
