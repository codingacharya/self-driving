"""Self-driving car with Q-learning: animated classroom simulator (Streamlit only).
Run: pip install -r requirements.txt  ->  streamlit run streamlit_app.py
"""
import json
import random
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Self-driving car: Q-learning", layout="wide")

# ---------------- Environment: 3-lane highway ----------------
LANES, ROAD, MAX_STEPS = 3, 60, 90
ACTIONS = ["Keep", "Accelerate", "Brake", "Lane left", "Lane right"]
GAP = ["close", "medium", "far"]
TRACKED = {
    "Car close ahead, both side lanes free, fast": (0, 1, 1, 2),
    "Car close ahead, left blocked, right free": (0, 0, 1, 1),
    "Open road ahead": (2, 1, 1, 1),
}


def describe(s):
    g, l, r, v = s
    return f"Gap: {GAP[g]} | Left: {'Free' if l else 'Blocked'} | Right: {'Free' if r else 'Blocked'} | Speed: {v + 1}"


class Highway:
    def reset(self, cars=None):
        self.x, self.lane, self.v, self.t = 0, 1, 2, 0
        self.cars = [c[:] for c in cars] if cars else \
            [[random.randrange(LANES), random.randrange(7, ROAD - 3)] for _ in range(7)]
        return self.state()

    def gap(self, lane):
        d = [c[1] - self.x for c in self.cars if c[0] == lane and c[1] > self.x]
        return min(d) if d else 99

    def side_free(self, lane):
        if lane < 0 or lane >= LANES:
            return 0
        return int(not any(c[0] == lane and -1 <= c[1] - self.x <= 3 for c in self.cars))

    def state(self):
        g = self.gap(self.lane)
        gb = 0 if g <= 2 else 1 if g <= 5 else 2
        return (gb, self.side_free(self.lane - 1), self.side_free(self.lane + 1), self.v - 1)

    def step(self, a, rewards):
        r = -0.5
        x0 = self.x
        old_cars = [c[:] for c in self.cars]

        if a == 1:
            self.v = min(3, self.v + 1)
        elif a == 2:
            self.v = max(1, self.v - 1)
        elif a in (3, 4):
            nl = self.lane + (-1 if a == 3 else 1)
            if 0 <= nl < LANES and self.side_free(nl):
                self.lane = nl
                r += rewards["lane_change"]
            else:
                r += rewards["lane_change"] - 5.0

        self.x += self.v
        for c in self.cars:
            c[1] += 1
        self.t += 1

        crash = any(c[0] == self.lane and old_c[1] >= x0 and c[1] <= self.x for c, old_c in zip(self.cars, old_cars))
        s_next = self.state()

        if crash:
            return s_next, r + rewards["crash"], True, "collision"
        if self.gap(self.lane) <= 1:
            r += rewards["tailgate"]
        
        r += self.v * rewards["progress"]

        if self.x >= ROAD:
            return s_next, r + rewards["goal"], True, "goal"
        if self.t >= MAX_STEPS:
            return s_next, r, True, "timeout"

        return s_next, r, False, ""


def pick(Q, s, eps):
    if random.random() < eps:
        return random.randrange(len(ACTIONS))
    q = Q[s]
    return int(random.choice(np.flatnonzero(q == q.max())))


# ---------------- Training Logic ----------------
def train(n, alpha, gamma, eps0, rewards, reset_q=False):
    if reset_q or "Q_table" not in st.session_state:
        Q = np.zeros((3, 2, 2, 3, len(ACTIONS)))
    else:
        Q = st.session_state.Q_table

    env, rows, mistakes = Highway(), [], []
    track = {k: [] for k in TRACKED}
    xs = []
    bar = st.progress(0.0, text="Training AI Driver...")

    for ep in range(n):
        s, total, done = env.reset(), 0, False
        eps = max(0.02, eps0 * (1 - ep / n))
        
        while not done:
            a = pick(Q, s, eps)
            s2, r, done, why = env.step(a, rewards)
            old_q = Q[s][a]
            
            max_q_next = 0 if done else Q[s2].max()
            Q[s][a] += alpha * (r + gamma * max_q_next - old_q)

            if why == "collision" and len(mistakes) < 8:
                mistakes.append({
                    "Episode": ep + 1, "Step": env.t, "Situation": describe(s),
                    "Action": ACTIONS[a], "Reward": round(r, 1),
                    "Q before": round(old_q, 2), "Q after": round(Q[s][a], 2)
                })

            s, total = s2, total + r

        rows.append((total, why, ACTIONS[a], eps))

        if ep % 20 == 0:
            xs.append(ep)
            for k, st_ in TRACKED.items():
                track[k].append(Q[st_].tolist())

        if ep % max(1, (n // 100)) == 0:
            bar.progress(ep / n)

    bar.empty()
    st.session_state.Q_table = Q

    df = pd.DataFrame(rows, columns=["reward", "outcome", "last_action", "eps"])
    df["chunk"] = df.index // max(1, (n // 10))
    g = df.groupby("chunk")
    prog = pd.DataFrame({
        "Goal reached %": g.outcome.apply(lambda o: 100 * (o == "goal").mean()),
        "Crashed %": g.outcome.apply(lambda o: 100 * (o == "collision").mean())
    })
    prog.index = (prog.index + 1) * (n // 10)
    reward = pd.DataFrame({"Average reward": g.reward.mean()})
    reward.index = prog.index

    k = max(20, n // 5)
    crash = df[df.outcome == "collision"]
    by_action = pd.DataFrame({
        "First 20% of session": crash.iloc[:0].last_action.value_counts() if crash.empty else
            df.iloc[:k][df.iloc[:k].outcome == "collision"].last_action.value_counts(),
        "Last 20% of session": df.iloc[-k:][df.iloc[-k:].outcome == "collision"].last_action.value_counts(),
    }).reindex(ACTIONS).fillna(0)

    # Evaluation phase
    ev, evr = Highway(), []
    for _ in range(200):
        e_s, e_done = ev.reset(), False
        e_tot = 0
        while not e_done:
            e_s, e_r, e_done, e_why = ev.step(pick(Q, e_s, 0), rewards)
            e_tot += e_r
        evr.append((e_why, e_tot))

    st.session_state.result = dict(
        Q=Q, alpha=alpha, gamma=gamma, n=n, rewards=rewards, prog=prog, reward=reward, mistakes=mistakes, by_action=by_action,
        xs=xs, track={k: pd.DataFrame(v, index=xs, columns=ACTIONS) for k, v in track.items()},
        overall=(round(100 * (df.outcome == "goal").mean()), round(100 * (df.outcome == "collision").mean())),
        final=(
            round(100 * sum(w == "goal" for w, _ in evr) / 200),
            round(100 * sum(w == "collision" for w, _ in evr) / 200),
            round(sum(t for _, t in evr) / 200, 1)
        )
    )
    st.session_state.play = None


def run_episode(Q, cars, rewards):
    e = Highway()
    s = e.reset(cars)
    snap = lambda a, r, tot, q, d: dict(
        x=e.x, lane=e.lane, v=e.v, cars=[c[:] for c in e.cars], a=a,
        r=round(r, 1), tot=round(tot, 1), q=[round(v, 1) for v in q], d=d
    )
    frames, tot, done, why = [snap(None, 0, 0, [0] * 5, "start")], 0, False, ""
    while not done:
        q = Q[s].tolist()
        a = pick(Q, s, 0)
        d = describe(s)
        s, r, done, why = e.step(a, rewards)
        tot += r
        frames.append(snap(a, r, tot, q, d))
    return dict(frames=frames, outcome=why)


def simulate():
    rewards = st.session_state.result["rewards"]
    base = Highway()
    base.reset()
    cars = base.cars
    st.session_state.play = dict(
        untrained=run_episode(np.zeros((3, 2, 2, 3, 5)), cars, rewards),
        trained=run_episode(st.session_state.result["Q"], cars, rewards)
    )


# ---------------- Interactive HTML, CSS & jQuery Component ----------------
HTML_UI = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<style>
  body { margin:0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; }
  .control-bar { display: flex; gap: 12px; align-items: center; background: #1e293b; padding: 10px 16px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #334155; }
  .btn { padding: 8px 16px; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; transition: all 0.2s ease; }
  .btn-primary { background: #0ea5e9; color: #fff; }
  .btn-primary:hover { background: #0284c7; }
  .btn-secondary { background: #475569; color: #fff; }
  .btn-secondary:hover { background: #334155; }
  select { padding: 6px 12px; background: #0f172a; color: #fff; border: 1px solid #475569; border-radius: 6px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 14px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }
  .card-header { display: flex; justify-content: space-between; align-items: center; font-size: 15px; font-weight: 700; margin-bottom: 8px; color: #38bdf8; }
  canvas { width: 100%; border-radius: 6px; display: block; background: #090d16; }
  .grid-layout { display: flex; gap: 12px; margin-top: 10px; }
  .log-box { flex: 1.3; height: 135px; overflow-y: auto; font-size: 12px; background: #0f172a; padding: 8px; border-radius: 6px; border: 1px solid #334155; }
  .q-box { flex: 1; font-size: 12px; background: #0f172a; padding: 8px; border-radius: 6px; border: 1px solid #334155; }
  .log-item { padding: 3px 0; border-bottom: 1px solid #1e293b; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; }
  .badge { color: #fff; border-radius: 4px; padding: 2px 6px; font-weight: bold; margin-right: 6px; font-size: 10px; text-transform: uppercase; }
  .q-row { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
  .q-label { width: 75px; font-weight: 600; color: #94a3b8; }
  .q-track { flex: 1; height: 12px; background: #334155; border-radius: 3px; overflow: hidden; position: relative; }
  .q-bar { height: 100%; border-radius: 3px; transition: width 0.1s ease; }
  .badge-status { background: #38bdf8; color: #0f172a; padding: 3px 8px; border-radius: 12px; font-size: 11px; }
</style>
</head>
<body>

<div class="control-bar">
  <button class="btn btn-primary" id="btn-play">Pause</button>
  <button class="btn btn-secondary" id="btn-restart">Restart</button>
  <span style="font-size: 13px; color: #94a3b8;">Speed:</span>
  <select id="sel-speed">
    <option value="900">Slow</option>
    <option value="550" selected>Normal</option>
    <option value="200">Fast</option>
  </select>
  <span style="font-size: 12px; color: #64748b; margin-left: auto;">Identical Traffic Seed Comparison</span>
</div>

<div id="sim-container"></div>

<script>
const D = __DATA__, ACT = D.actions, ROAD = D.road;
const ACTION_COLORS = ["#64748b", "#22c55e", "#ef4444", "#3b82f6", "#a855f7"];
let speed = 550, playing = true, t = 0, last = performance.now();

function createPanel(key, title) {
  const html = `
    <div class="card" id="panel-${key}">
      <div class="card-header">
        <span>${title}</span>
        <span class="badge-status status-text">Initializing...</span>
      </div>
      <canvas width="960" height="210"></canvas>
      <div class="grid-layout">
        <div class="log-box"></div>
        <div class="q-box"></div>
      </div>
    </div>`;
  $('#sim-container').append(html);
  const $p =$(`#panel-${key}`);
  return {
    f: D[key].frames, out: D[key].outcome,
    ctx: $p.find('canvas')[0].getContext('2d'),
    $log:$p.find('.log-box'), $q:$p.find('.q-box'),
    $status:$p.find('.status-text'), shown: 0
  };
}

const Panels = [
  createPanel('untrained', 'Before Training: Random Exploration Policy'),
  createPanel('trained', 'After Training: Learned Optimal Policy')
];

function drawCar(ctx, px, lane, color, isSelf) {
  const y = lane * 70 + 16, w = 36, h = 38;
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.roundRect(px + 3, y, w, h, 6); ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,0.7)';
  ctx.fillRect(px + w - 9, y + 6, 5, h - 12);
  if (isSelf) { ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke(); }
}

function lerp(a, b, u) { return a + (b - a) * u; }

function render(p) {
  const n = p.f.length, k = Math.min(Math.floor(t), n - 1);
  const f = p.f[k], g = p.f[Math.min(k + 1, n - 1)];
  let u = k >= n - 1 ? 0 : t - k;
  u = u * u * (3 - 2 * u);

  const ex = lerp(f.x, g.x, u), el = lerp(f.lane, g.lane, u), cam = ex - 4, ctx = p.ctx;
  ctx.fillStyle = '#090d16'; ctx.fillRect(0, 0, 960, 210);

  // Road markings
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 2; ctx.setLineDash([18, 16]);
  ctx.lineDashOffset = (cam * 40) % 34;
  for (let l = 1; l < 3; l++) { ctx.beginPath(); ctx.moveTo(0, l * 70); ctx.lineTo(960, l * 70); ctx.stroke(); }
  ctx.setLineDash([]);

  // Finish Line
  const fx = (ROAD - cam) * 40;
  if (fx < 960) {
    for (let r = 0; r < 21; r++) {
      ctx.fillStyle = r % 2 ? '#1e293b' : '#f8fafc';
      ctx.fillRect(fx, r * 10, 10, 10);
      ctx.fillRect(fx + 10, (r + 1) % 2 ? '#1e293b' : '#f8fafc', 10, 10);
    }
  }

  // Traffic & Ego Vehicle
  f.cars.forEach((cc, i) => {
    const px = (lerp(cc[1], g.cars[i][1], u) - cam) * 40;
    if (px > -40 && px < 960) drawCar(ctx, px, cc[0], '#f43f5e', false);
  });
  drawCar(ctx, 4 * 40, el, '#0ea5e9', true);

  // Dynamic Log & Q-value Update via jQuery
  const dec = k < n - 1 ? k + 1 : n - 1;
  while (p.shown < dec) {
    p.shown++;
    const qf = p.f[p.shown];
    const logEntry = `<div class="log-item">
      <span class="badge" style="background:${ACTION_COLORS[qf.a]}">${ACT[qf.a]}</span>
      Step ${p.shown}: ${qf.d} | Reward: <b>${qf.r}</b>
    </div>`;
    p.$log.append(logEntry);
    p.$log.scrollTop(p.$log[0].scrollHeight);
  }

  if (dec >= 1) {
    const qf = p.f[dec], minQ = Math.min(...qf.q), maxQ = Math.max(...qf.q);
    const getW = v => maxQ > minQ ? 8 + 92 * (v - minQ) / (maxQ - minQ) : 50;
    
    let qHtml = '<div style="font-weight:bold;margin-bottom:4px;color:#94a3b8">Action Q-Values</div>';
    ACT.forEach((aName, idx) => {
      const isChosen = idx === qf.a;
      qHtml += `<div class="q-row">
        <span class="q-label">${aName}</span>
        <div class="q-track">
          <div class="q-bar" style="width:${getW(qf.q[idx])}%; background:${isChosen ? '#f59e0b' : '#0ea5e9'}"></div>
        </div>
        <span style="width:35px;text-align:right;font-size:11px">${qf.q[idx]}</span>
      </div>`;
    });
    p.$q.html(qHtml);
  }

  const cur = p.f[dec];
  p.$status.html(`Step ${dec} | Speed: ${cur.v} | Total Reward: ${cur.tot}`);

  if (k >= n - 1 && t >= n - 1) {
    const ok = p.out === 'goal';
    ctx.fillStyle = ok ? 'rgba(34, 197, 94, 0.35)' : 'rgba(239, 68, 68, 0.45)';
    ctx.fillRect(0, 0, 960, 210);
    ctx.fillStyle = '#ffffff'; ctx.font = 'bold 42px system-ui'; ctx.textAlign = 'center';
    ctx.fillText(ok ? 'GOAL REACHED!' : p.out === 'collision' ? 'COLLISION!' : 'TIME OUT', 480, 120);
    ctx.textAlign = 'left';
  }
}

function animLoop(now) {
  const dt = now - last; last = now;
  if (playing) t += dt / speed;
  const maxFrames = Math.max(...Panels.map(p => p.f.length)) + 1;
  if (t > maxFrames) { t = maxFrames; playing = false; $('#btn-play').text('Replay'); }
  Panels.forEach(render);
  requestAnimationFrame(animLoop);
}

$('#btn-play').on('click', function() {
  if ($(this).text() === 'Replay') restart();
  else { playing = !playing; $(this).text(playing ? 'Pause' : 'Resume'); }
});

function restart() {
  t = 0; playing = true; $('#btn-play').text('Pause');
  Panels.forEach(p => { p.shown = 0; p.$log.empty(); });
}

$('#btn-restart').on('click', restart);
$('#sel-speed').on('change', function() { speed = +$(this).val(); });

requestAnimationFrame(animLoop);
</script>
</body>
</html>
"""

# ---------------- UI & Control Logic ----------------
st.title("Self-Driving Highway Simulator with Q-Learning")
st.caption("Interactive Reinforcement Learning Environment with customizable rewards and real-time state-action analysis.")

with st.sidebar:
    st.header("1. Hyperparameters")
    n = st.slider("Episodes", 200, 6000, 2000, 200)
    alpha = st.slider("Learning Rate α", 0.01, 0.5, 0.10, 0.01)
    gamma = st.slider("Discount γ", 0.5, 0.99, 0.95, 0.01)
    eps0 = st.slider("Initial Exploration ε", 0.05, 1.0, 0.5, 0.05)

    st.header("2. Custom Reward Model")
    r_crash = st.slider("Collision Penalty", -200, -10, -100, 10)
    r_goal = st.slider("Goal Reward", 10, 100, 50, 5)
    r_tailgate = st.slider("Tailgating Penalty", -20, 0, -5, 1)
    r_lane = st.slider("Lane Switch Cost", -10, 0, -1, 1)
    r_progress = st.slider("Speed Progress Multiplier", 0.5, 3.0, 1.0, 0.1)

    rewards = {
        "crash": r_crash, "goal": r_goal,
        "tailgate": r_tailgate, "lane_change": r_lane,
        "progress": r_progress
    }

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Continue Training", type="primary"):
            train(n, alpha, gamma, eps0, rewards, reset_q=False)
    with col_btn2:
        if st.button("Reset & Train"):
            train(n, alpha, gamma, eps0, rewards, reset_q=True)

R = st.session_state.get("result")
if not R:
    st.info("Configure the hyperparameters & rewards in the sidebar, then click **Continue Training** or **Reset & Train** to start.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Watch Simulation", "Learning Analysis", "Full Q-Table Inspector", "Training Statistics"])

with tab1:
    if st.button("Run Simulation with Fresh Random Traffic") or st.session_state.get("play") is None:
        simulate()
    data = dict(actions=ACTIONS, road=ROAD, **st.session_state.play)
    page = HTML_UI.replace("__DATA__", json.dumps(data))
    if hasattr(st, "iframe"):
        st.iframe(page, height=1050)
    else:
        components.html(page, height=1050, scrolling=True)

with tab2:
    st.subheader("Mistake & Penalty Breakdown")
    if R["mistakes"]:
        m = R["mistakes"][0]
        new_val = m["Q before"] + R["alpha"] * (m["Reward"] - m["Q before"])
        st.markdown(
            f"In **episode {m['Episode']}** (step {m['Step']}) the car encountered state: *{m['Situation']}*. "
            f"It picked **{m['Action']}** and crashed, receiving reward **{m['Reward']}**.\n\n"
            f"`Q_new = Q_old + α × (reward − Q_old) = {m['Q before']} + {R['alpha']} × ({m['Reward']} − {m['Q before']}) = {new_val:.2f}`"
        )
        st.dataframe(pd.DataFrame(R["mistakes"]), hide_index=True)
    else:
        st.write("No crash mistakes recorded in this session.")

    st.subheader("Tracked Q-Value Convergence")
    sit = st.selectbox("Select State Situation", list(TRACKED))
    st.line_chart(R["track"][sit], x_label="Training Episodes", y_label="Action Q-Values")

with tab3:
    st.subheader("Complete State-Space Q-Table Heatmap")
    st.caption("Examine the learned Q-values and optimal policy across all 36 possible discrete states.")
    
    Q_arr = st.session_state.Q_table
    q_records = []
    
    for g_idx, g_name in enumerate(GAP):
        for l_idx, l_name in enumerate(["Blocked", "Free"]):
            for r_idx, r_name in enumerate(["Blocked", "Free"]):
                for v_idx in range(3):
                    state_tuple = (g_idx, 1 if l_name == "Free" else 0, 1 if r_name == "Free" else 0, v_idx)
                    q_vals = Q_arr[state_tuple]
                    best_act = ACTIONS[np.argmax(q_vals)]
                    q_records.append({
                        "Gap": g_name, "Left": l_name, "Right": r_name, "Speed": v_idx + 1,
                        "Optimal Action": best_act,
                        "Keep": round(q_vals[0], 2), "Accelerate": round(q_vals[1], 2),
                        "Brake": round(q_vals[2], 2), "Lane Left": round(q_vals[3], 2),
                        "Lane Right": round(q_vals[4], 2)
                    })
    
    df_q = pd.DataFrame(q_records)
    st.dataframe(df_q, use_container_width=True, hide_index=True)

with tab4:
    a, b, c, d = st.columns(4)
    a.metric("Episodes Trained", R["n"])
    b.metric("Test Goal Success Rate", f"{R['final'][0]}%")
    c.metric("Test Collision Rate", f"{R['final'][1]}%")
    d.metric("Test Avg Reward", R["final"][2])

    l, r = st.columns(2)
    l.caption("Goal Success & Collision Rates (%)")
    l.line_chart(R["prog"], x_label="Episode")
    r.caption("Mean Reward per Interval")
    r.line_chart(R["reward"], x_label="Episode")