"""
Project Oracle Final — The Pure Predictive World Model
================================================
Scientific Hypothesis:
An agent that continuously predicts future observations and updates its internal 
belief state using prediction error should gradually learn the dynamics of an 
environment, without any RL rewards, explicit planning, or supervised labels.

Core principles of the Final Version:
  1. No planning. The controller simply moves the paddle to predicted ball_y.
  2. The network observes 5D [ball_x, ball_y, paddle_y, action, visible].
  3. The network predicts 4D [ball_x, ball_y, paddle_y, visible]. It does NOT predict action.
  4. Constant Learning Rate. No confounding adaptive scaling.
  5. Read-only t+5 horizon visualization to verify physics modeling vs interpolation.
"""

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import argparse, csv, os, time
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG (FROZEN)
# ─────────────────────────────────────────────
W, H        = 800, 600
PADDLE_W    = 12
PADDLE_H    = 80
BALL_R      = 8
FPS         = 60

# Brain dims
OBS_DIM     = 5       # [ball_x, ball_y, paddle_y, action, visible]
PRED_DIM    = 4       # [ball_x, ball_y, paddle_y, visible]
HIDDEN_DIM  = 96      # custom cell hidden state

# Learning — strict and constant
BASE_LR                 = 3e-4
WEIGHT_UPDATE_SURPRISE  = 0.05    # surprise threshold to trigger weight update at all
WEIGHT_UPDATE_INTERVAL  = 30      # minimum frames between weight updates
MIN_STEPS_BEFORE_BACKPROP = 200   # don't touch weights until brain has some experience
PRED_HORIZON            = 5       # For read-only visualization

# Hidden state snapshots
SNAPSHOT_EVERY  = 500

# Physics change interval
PHYSICS_CHANGE_EVERY = 1500

# Colors
BG       = (10, 10, 20)
BALL_C   = (240, 200, 80)
PADDLE_C = (80, 200, 240)
PRED_C   = (240, 80, 80)
TEXT_C   = (200, 200, 200)
GRAPH_BG = (20, 20, 35)
GREEN    = (80, 220, 100)
RED      = (220, 80, 80)
YELLOW   = (240, 200, 80)
CYAN     = (80, 220, 220)
PURPLE   = (180, 80, 220)
OCCLUDE  = (40, 40, 80)      # tunnel color


# ─────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────
class Logger:
    LOG_EVERY = 100

    def __init__(self, experiment: int, noise_std: float):
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/snapshots", exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_exp{experiment}"

        def _open(name, header):
            fh = open(f"logs/{name}_{self.run_id}.csv", "w", newline="")
            w  = csv.writer(fh)
            w.writerow(header)
            return fh, w

        # Granular logging replacing the single scalar loss
        self._fh_f, self._w_f = _open("frames", [
            "frame", "occluded",
            "err_bx", "err_by", "err_py", "err_vis",
            "h_norm", "belief_update_mag",
            "weight_updates", 
            "hit_ratio", "total_hits", "total_misses",
            "gravity", "bounce", "ball_speed",
        ])
        self._fh_r, self._w_r = _open("rallies", [
            "rally_num", "frame", "outcome", "hit_ratio",
        ])
        self._fh_e, self._w_e = _open("events", [
            "frame", "event", "detail",
        ])
        self._rally_num = 0
        print(f"[Logger] logs/*_{self.run_id}.csv")

    def log_frame(self, frame, agent, env, occluded):
        if frame % self.LOG_EVERY != 0:
            return
        
        ratio = env.score / max(env.score + env.misses, 1)
        errs = agent.last_error_vector
        
        self._w_f.writerow([
            frame, int(occluded),
            round(errs[0], 6), round(errs[1], 6), round(errs[2], 6), round(errs[3], 6),
            round(agent.last_h_norm, 6), round(agent.last_belief_mag, 6),
            agent.weight_update_count,
            round(ratio, 4), env.score, env.misses,
            round(env.gravity, 6), round(env.bounce, 6), round(env.ball_speed, 4),
        ])

    def log_rally(self, frame, outcome, agent, env):
        self._rally_num += 1
        ratio = env.score / max(env.score + env.misses, 1)
        self._w_r.writerow([self._rally_num, frame, outcome, round(ratio,4)])

    def log_event(self, frame, event, detail=""):
        self._w_e.writerow([frame, event, detail])

    def save_snapshot(self, frame, hidden: np.ndarray):
        path = f"logs/snapshots/h_{self.run_id}_f{frame:07d}.npy"
        np.save(path, hidden)

    def flush(self):
        self._fh_f.flush(); self._fh_r.flush(); self._fh_e.flush()

    def close(self):
        self._fh_f.close(); self._fh_r.close(); self._fh_e.close()
        print(f"[Logger] Saved → logs/*_{self.run_id}.csv")


# ─────────────────────────────────────────────
# PONG ENVIRONMENT
# ─────────────────────────────────────────────
class Tunnel:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def contains(self, bx, by):
        return self.rect.collidepoint(bx, by)

    @classmethod
    def random(cls):
        w = np.random.randint(80, 220)
        h = np.random.randint(60, 200)
        x = np.random.randint(100, W - w - 60)
        y = np.random.randint(20, H - h - 20)
        return cls(x, y, w, h)


class PongEnv:
    def __init__(self, experiment=1, noise_std=0.0):
        self.experiment = experiment
        self.noise_std = noise_std
        self.reset_physics()
        self.tunnel: Tunnel | None = None
        self.last_action = 0.5
        self.reset()

    def reset_physics(self):
        self.gravity    = 0.0
        self.bounce     = 1.0
        self.friction   = 1.0
        self.ball_speed = 5.0

    def mutate_physics(self):
        self.gravity    = np.random.uniform(-0.05, 0.05)
        self.bounce     = np.random.uniform(0.9, 1.1)
        self.friction   = np.random.uniform(0.998, 1.002)
        self.ball_speed = np.random.uniform(4.0, 7.0)

    def spawn_tunnel(self):
        self.tunnel = Tunnel.random()

    def clear_tunnel(self):
        self.tunnel = None

    def reset(self):
        self.ball_x = W / 2
        self.ball_y = H / 2
        angle = np.random.uniform(-np.pi/4, np.pi/4)
        direction = np.random.choice([-1, 1])
        self.vx = self.ball_speed * np.cos(angle) * direction
        self.vy = self.ball_speed * np.sin(angle)
        self.paddle_y = H / 2
        self.score  = 0
        self.misses = 0
        self.last_action = 0.5
        return self.observe(occluded=False)

    def is_occluded(self):
        return self.tunnel is not None and self.tunnel.contains(self.ball_x, self.ball_y)

    def observe(self, occluded: bool):
        # 5D Observation: [ball_x, ball_y, paddle_y, action, visible]
        visible = 0.0 if occluded else 1.0
        bx = 0.0 if occluded else self.ball_x / W
        by = 0.0 if occluded else self.ball_y / H
        py = self.paddle_y / H
        
        # Exp 4: Sensor Noise
        if self.noise_std > 0 and not occluded:
            bx = np.clip(bx + np.random.normal(0, self.noise_std), 0, 1)
            by = np.clip(by + np.random.normal(0, self.noise_std), 0, 1)
        
        return np.array([bx, by, py, self.last_action, visible], dtype=np.float32)

    def step(self, paddle_action):
        self.last_action = paddle_action
        target_y = float(paddle_action) * H
        self.paddle_y += np.clip(target_y - self.paddle_y, -6, 6)
        self.paddle_y  = np.clip(self.paddle_y, PADDLE_H//2, H - PADDLE_H//2)

        self.vy += self.gravity
        self.vx *= self.friction
        self.vy *= self.friction
        
        old_x = self.ball_x
        self.ball_x += self.vx
        self.ball_y += self.vy

        missed = False

        # Floor / Ceiling
        if self.ball_y - BALL_R <= 0:
            self.ball_y = BALL_R;       self.vy =  abs(self.vy) * self.bounce
        if self.ball_y + BALL_R >= H:
            self.ball_y = H - BALL_R;   self.vy = -abs(self.vy) * self.bounce
            
        # Exp 5: Invisible Wall in the dead center of the screen
        if self.experiment == 5:
            wall_x = W / 2
            if old_x < wall_x and self.ball_x + BALL_R >= wall_x:
                self.ball_x = wall_x - BALL_R
                self.vx = -abs(self.vx) * self.bounce
            elif old_x > wall_x and self.ball_x - BALL_R <= wall_x:
                self.ball_x = wall_x + BALL_R
                self.vx = abs(self.vx) * self.bounce
        
        # Right Wall (safe bounce)
        if self.ball_x + BALL_R >= W:
            self.ball_x = W - BALL_R;   self.vx = -abs(self.vx) * self.bounce

        # Paddle Collision (Left side)
        pl, pr = 40, 40 + PADDLE_W
        if (self.ball_x - BALL_R <= pr and
                self.ball_x + BALL_R >= pl and
                abs(self.ball_y - self.paddle_y) <= PADDLE_H // 2):
            self.ball_x = pr + BALL_R
            self.vx = abs(self.vx) * self.bounce
            self.score += 1
        elif self.ball_x - BALL_R <= 0:
            missed = True
            self.misses += 1
            self.reset()

        occluded = self.is_occluded()
        return self.observe(occluded), missed, occluded


# ─────────────────────────────────────────────
# CUSTOM RECURRENT CELL
# ─────────────────────────────────────────────
class PredictiveCell(nn.Module):
    def __init__(self, obs_dim, err_dim, hidden_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_x = nn.Linear(obs_dim,    hidden_dim, bias=False)
        self.W_e = nn.Linear(err_dim,    hidden_dim, bias=False)
        self.b   = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, obs: torch.Tensor, h: torch.Tensor, error: torch.Tensor):
        pre = self.W_h(h) + self.W_x(obs) + self.W_e(error)
        return torch.tanh(self.norm(pre + self.b))


# ─────────────────────────────────────────────
# PREDICTIVE BRAIN
# ─────────────────────────────────────────────
class PredictiveBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Input 5D observation + 4D error to output 96D hidden
        self.cell = PredictiveCell(OBS_DIM, PRED_DIM, HIDDEN_DIM)
        # Decode 96D hidden to 4D prediction (x, y, paddle, visible)
        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, PRED_DIM),
            nn.Sigmoid(),
        )

    def predict_one(self, obs: torch.Tensor, h: torch.Tensor,
                    error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new_h = self.cell(obs, h, error)
        pred  = self.decoder(new_h)
        return pred, new_h


# ─────────────────────────────────────────────
# PREDICTIVE AGENT
# ─────────────────────────────────────────────
class PredictiveAgent:
    def __init__(self, noise_std=0.0):
        self.noise_std  = noise_std
        self.brain      = PredictiveBrain()
        self.optimizer  = optim.Adam(self.brain.parameters(), lr=BASE_LR) # Constant LR
        
        self.h = torch.zeros(1, HIDDEN_DIM)
        self.last_pred_t1: torch.Tensor | None = None
        
        # Logging metrics
        self.last_surprise     = 0.0
        self.last_error_vector = [0.0, 0.0, 0.0, 0.0]
        self.last_h_norm       = 0.0
        self.last_belief_mag   = 0.0
        self.horizon_preds     = [] # stores read-only t+1..t+5 for visual debug
        
        self.error_history     = deque(maxlen=500)
        self.weight_update_count = 0
        self._last_weight_update_frame = -WEIGHT_UPDATE_INTERVAL
        self._frame            = 0

        self._batch_obs: list[np.ndarray] = []
        self._BATCH_CAP = 64

    def step(self, obs: np.ndarray, occluded: bool) -> float:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        # 1. Compute prediction error for BELIEF UPDATE
        if self.last_pred_t1 is not None:
            # Reality is first 4 dims, prediction is 4 dims
            raw_err = obs_t[:, :4] - self.last_pred_t1
            
            # Mask out x and y errors if occluded (so we don't punish the network for reality being invisible)
            # The visibility flag error remains active and backpropagates!
            mask = torch.ones_like(raw_err)
            mask[:, 0:2] = obs_t[:, 4:5] 
            
            error_t = (raw_err * mask).detach() # Detached to prevent massive computation graph memory leak
            
            self.last_error_vector = error_t.abs().squeeze().tolist()
            surprise = float(error_t.norm().item())
        else:
            error_t = torch.zeros(1, PRED_DIM)
            surprise = 0.0
            self.last_error_vector = [0.0]*4

        self.last_surprise = surprise
        self.error_history.append(surprise)

        # 2. Step the world model forward (Belief Update)
        with torch.no_grad():
            pred, new_h_t = self.brain.predict_one(obs_t, self.h, error_t)
            
            # Read-only Imagination: Roll forward PRED_HORIZON steps for visualization
            img_h = new_h_t
            img_obs = obs_t.clone()
            self.horizon_preds = [pred.squeeze().numpy()]
            
            for _ in range(1, PRED_HORIZON):
                err_zero = torch.zeros(1, PRED_DIM)
                img_pred, img_h = self.brain.predict_one(img_obs, img_h, err_zero)
                self.horizon_preds.append(img_pred.squeeze().numpy())
                
                # Construct fake next observation from this prediction
                next_obs = torch.zeros(1, 5)
                next_obs[:, :4] = img_pred
                next_obs[:, 4]  = obs_t[:, 3] # paddle action remains constant in imagination
                img_obs = next_obs

        self.last_belief_mag = float((new_h_t - self.h).norm().item())
        self.h = new_h_t.detach() 
        self.last_h_norm = float(self.h.norm().item())
        
        self.last_pred_t1 = pred.detach()

        # Always accumulate observations for BPTT
        self._batch_obs.append(obs.copy())
        if len(self._batch_obs) > self._BATCH_CAP:
            self._batch_obs.pop(0)

        # 3. Slow learning (Weight Update via BPTT)
        frames_since_update = self._frame - self._last_weight_update_frame
        if (self._frame >= MIN_STEPS_BEFORE_BACKPROP and
                surprise > WEIGHT_UPDATE_SURPRISE and
                frames_since_update >= WEIGHT_UPDATE_INTERVAL and
                len(self._batch_obs) > 4):
            self._weight_update()
            self._last_weight_update_frame = self._frame

        # 4. "Stupid" Controller - just move to predicted ball_y
        action = self._stupid_controller()

        self._frame += 1
        return action

    def _weight_update(self):
        obs_seq = torch.FloatTensor(np.stack(self._batch_obs))
        seq_len = len(obs_seq)
        
        h_t = torch.zeros(1, HIDDEN_DIM)
        err_t = torch.zeros(1, PRED_DIM)
        total_loss = 0.0
        
        for t in range(seq_len - 1):
            obs_t = obs_seq[t : t+1]
            target_obs = obs_seq[t+1 : t+2]
            
            target_4d = target_obs[:, :4]
            target_vis = target_obs[:, 4:5]
            
            pred, h_t = self.brain.predict_one(obs_t, h_t, err_t)
            
            raw_err = target_4d - pred
            
            mask = torch.ones_like(raw_err)
            mask[:, 0:2] = target_vis
            
            # The gradient flows entirely through this error term back into the cell!
            err_t = raw_err * mask 
            
            total_loss = total_loss + (err_t ** 2).sum()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 0.5)
        self.optimizer.step()
        
        self.weight_update_count += 1
        self._batch_obs.clear()

    def _stupid_controller(self) -> float:
        """
        The controller is deliberately stupid to isolate the hypothesis.
        Move paddle to exactly where the brain predicts the ball will be on the y-axis.
        """
        if self.last_pred_t1 is not None:
            return np.clip(float(self.last_pred_t1[0, 1].item()), 0.0, 1.0)
        return 0.5

    def reset_belief(self):
        self.h = torch.zeros(1, HIDDEN_DIM)
        self.last_pred_t1 = None
        self.last_error_vector = [0.0]*4
        self.horizon_preds.clear()
        self._batch_obs.clear()
        self._frame = 0

    def hidden_snapshot(self) -> np.ndarray:
        return self.h.detach().squeeze(0).numpy().copy()


# ─────────────────────────────────────────────
# GRAPH HELPER
# ─────────────────────────────────────────────
def draw_graph(surface, history, rect, color, max_val=1.0, vlines=None):
    pygame.draw.rect(surface, GRAPH_BG, rect)
    pygame.draw.rect(surface, (40, 40, 60), rect, 1)
    data = list(history)
    if len(data) < 2:
        return
    n = len(data)
    rw, rh = rect.width - 4, rect.height - 4
    x0, y0 = rect.x + 2, rect.y + 2
    max_v = max(max(data) * 1.1, 1e-6) if max_val is None else max_val
    pts = []
    for i, v in enumerate(data):
        px = x0 + int(i / (n - 1) * rw)
        py = y0 + rh - int(min(v / max_v, 1.0) * rh)
        pts.append((px, py))
    pygame.draw.lines(surface, color, False, pts, 2)
    if vlines:
        for vf in vlines:
            rel = len(data) - 1 - vf
            if 0 <= rel < n:
                idx = n - 1 - rel
                gx  = x0 + int(idx / max(n-1,1) * rw)
                pygame.draw.line(surface, YELLOW, (gx, rect.y), (gx, rect.y+rect.height), 1)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def run(experiment: int = 1, noise_std: float = 0.0, physics_changes: bool = False):
    pygame.init()
    screen = pygame.display.set_mode((W + 300, H))
    pygame.display.set_caption("Project Oracle Final — Pure Predictive Brain")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 13)
    font_l = pygame.font.SysFont("monospace", 15, bold=True)

    if experiment == 4:
        noise_std = max(noise_std, 0.05)

    env    = PongEnv(experiment=experiment, noise_std=noise_std)
    agent  = PredictiveAgent(noise_std=noise_std)
    logger = Logger(experiment=experiment, noise_std=noise_std)

    if experiment == 3:
        physics_changes = True

    tunnel_timer    = 0
    TUNNEL_INTERVAL = 400    
    TUNNEL_DURATION = 180    

    obs = env.reset()
    frame = 0
    physics_event_log: list[int] = []
    occlusion_log: list[int]     = []
    running   = True
    paused    = False
    show_help = True
    occluded  = False
    _last_score = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    obs = env.reset()
                    env.clear_tunnel()
                    agent.reset_belief()
                    frame = 0
                    physics_event_log.clear()
                    occlusion_log.clear()
                if event.key == pygame.K_h:
                    show_help = not show_help
                if event.key == pygame.K_p:
                    env.mutate_physics()
                    physics_event_log.append(frame)
                    logger.log_event(frame, "physics_manual",
                        f"g={env.gravity:.4f} b={env.bounce:.4f} s={env.ball_speed:.2f}")
                if event.key == pygame.K_t:
                    if env.tunnel is None:
                        env.spawn_tunnel()
                        occlusion_log.append(frame)
                        logger.log_event(frame, "tunnel_spawned_manual", "")
                    else:
                        env.clear_tunnel()
                        logger.log_event(frame, "tunnel_cleared", "")

        if paused:
            clock.tick(FPS); continue

        if physics_changes and frame > 0 and frame % PHYSICS_CHANGE_EVERY == 0:
            env.mutate_physics()
            physics_event_log.append(frame)
            logger.log_event(frame, "physics_auto",
                f"g={env.gravity:.4f} b={env.bounce:.4f} s={env.ball_speed:.2f}")

        if experiment == 2:
            tunnel_timer += 1
            if tunnel_timer % TUNNEL_INTERVAL == 0:
                env.spawn_tunnel()
                occlusion_log.append(frame)
                logger.log_event(frame, "tunnel_auto", "")
            if tunnel_timer % TUNNEL_INTERVAL == TUNNEL_DURATION:
                env.clear_tunnel()

        action   = agent.step(obs, occluded)
        obs, missed, occluded = env.step(action)

        if frame % SNAPSHOT_EVERY == 0:
            snap = agent.hidden_snapshot()
            logger.save_snapshot(frame, snap)

        logger.log_frame(frame, agent, env, occluded)
        if missed:
            logger.log_rally(frame, "miss", agent, env)
        if env.score != _last_score:
            logger.log_rally(frame, "hit", agent, env)
        _last_score = env.score
        if frame % 300 == 0:
            logger.flush()

        frame += 1

        screen.fill(BG)

        # Court divider
        for y in range(0, H, 20):
            pygame.draw.rect(screen, (40, 40, 60), (W//2, y, 3, 10))
            
        # Exp 5: Invisible Wall visualization (faint so we know it's there)
        if experiment == 5:
            pygame.draw.line(screen, (100, 30, 30), (W//2, 0), (W//2, H), 2)

        if env.tunnel:
            pygame.draw.rect(screen, OCCLUDE, env.tunnel.rect)
            label = font.render("TUNNEL", True, (80, 80, 120))
            lx    = env.tunnel.rect.centerx - label.get_width() // 2
            ly    = env.tunnel.rect.centery - label.get_height() // 2
            screen.blit(label, (lx, ly))

        if not occluded:
            pygame.draw.circle(screen, BALL_C, (int(env.ball_x), int(env.ball_y)), BALL_R)
        else:
            pygame.draw.circle(screen, (80, 60, 30),
                               (int(env.ball_x), int(env.ball_y)), BALL_R, 1)

        # Draw Prediction Horizon t+1 to t+5
        for i, pred in enumerate(agent.horizon_preds):
            alpha = 1.0 - i / (PRED_HORIZON + 1)
            c = tuple(int(v * alpha) for v in PRED_C)
            r = max(2, BALL_R // 2 - i)
            px = int(pred[0] * W)
            py = int(pred[1] * H)
            pygame.draw.circle(screen, c, (px, py), r)
            if i == 0:
                bx = int(env.ball_x) if not occluded else px
                by = int(env.ball_y) if not occluded else py
                pygame.draw.line(screen, PRED_C, (bx, by), (px, py), 1)

        pygame.draw.rect(screen, PADDLE_C,
                         pygame.Rect(40, int(env.paddle_y)-PADDLE_H//2, PADDLE_W, PADDLE_H),
                         border_radius=4)

        sx = W + 8
        screen.fill((15, 15, 25), (W, 0, 300, H))

        def txt(text, x, y, color=TEXT_C, f=font):
            screen.blit(f.render(text, True, color), (x, y))

        txt("PROJECT ORACLE FINAL", sx, 8, YELLOW, font_l)
        txt("Pure World Model", sx, 26, (140, 140, 180))

        exp_labels = {
            1: "Exp 1: Static Physics",
            2: "Exp 2: Occlusion / Tunnel",
            3: "Exp 3: Physics Changes",
            4: "Exp 4: Sensor Noise",
            5: "Exp 5: Hidden Obstacle",
        }
        txt(exp_labels.get(experiment, f"Exp {experiment}"), sx, 44, GREEN)
        txt("─"*28, sx, 60, (50,50,70))

        ratio = env.score / max(env.score + env.misses, 1)
        txt(f"Frame        : {frame:>6}", sx, 74)
        txt(f"Score        : {env.score:>6}", sx, 90,  GREEN)
        txt(f"Misses       : {env.misses:>6}", sx, 106, RED)
        txt(f"Hit ratio    : {ratio:>6.1%}", sx, 122,
            GREEN if ratio > 0.6 else (YELLOW if ratio > 0.3 else RED))

        txt("─"*28, sx, 138, (50,50,70))
        txt("BELIEF STATE", sx, 152, YELLOW)

        occ_color = RED if occluded else GREEN
        txt(f"Occluded     : {'YES' if occluded else 'no':>6}", sx, 168, occ_color)
        txt(f"h_norm       : {agent.last_h_norm:>6.4f}", sx, 184)
        txt(f"belief_shift : {agent.last_belief_mag:>6.4f}", sx, 200, CYAN)

        avg_err = np.mean(list(agent.error_history)[-50:]) if agent.error_history else 0
        txt(f"Avg surp(50) : {avg_err:>6.4f}", sx, 216)
        txt(f"Weight upds  : {agent.weight_update_count:>6}", sx, 232)

        txt("─"*28, sx, 248, (50,50,70))
        txt("PREDICTION ERRORS", sx, 262, YELLOW)
        errs = agent.last_error_vector
        txt(f"ball_x       : {errs[0]:.4f}", sx, 278, RED if errs[0]>0.05 else GREEN)
        txt(f"ball_y       : {errs[1]:.4f}", sx, 294, RED if errs[1]>0.05 else GREEN)
        txt(f"paddle_y     : {errs[2]:.4f}", sx, 310, RED if errs[2]>0.05 else GREEN)
        txt(f"visibility   : {errs[3]:.4f}", sx, 326, RED if errs[3]>0.05 else GREEN)

        txt("─"*28, sx, 342, (50,50,70))
        txt("TOTAL SURPRISE", sx, 356, YELLOW)
        draw_graph(screen, agent.error_history,
                   pygame.Rect(sx, 372, 280, 120), RED, max_val=0.3,
                   vlines=[frame - f for f in physics_event_log + occlusion_log])

        txt("─"*28, sx, 502, (50,50,70))
        if show_help:
            txt("[T] spawn/clear tunnel", sx, 516, PURPLE)
            txt("[P] mutate physics", sx, 532, (100,100,130))
            txt("[SPACE] pause  [R] reset  [Q] quit", sx, 548, (100,100,130))
            txt("[H] toggle help", sx, 564, (100,100,130))

        pygame.draw.circle(screen, BALL_C,  (sx+6,  H-16), 5)
        txt(" real ball", sx+12, H-23, (160,160,160))
        pygame.draw.circle(screen, PRED_C,  (sx+80, H-16), 4)
        txt(" t+1..t+5", sx+86, H-23, (160,160,160))

        pygame.display.flip()
        clock.tick(FPS)

    logger.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project Oracle Final — Pure Predictive Brain")
    parser.add_argument("--experiment", "-e", type=int, default=1,
        choices=[1,2,3,4,5],
        help=("1=static  2=occlusion/tunnel  3=physics-changes  "
              "4=sensor-noise  5=hidden-obstacle"))
    parser.add_argument("--noise", "-n", type=float, default=0.0,
        help="Observation noise std (0=none, 0.05=moderate)")
    parser.add_argument("--physics-changes", action="store_true",
        help="Force physics mutations regardless of experiment")
    args = parser.parse_args()
    run(experiment=args.experiment, noise_std=args.noise,
        physics_changes=args.physics_changes)