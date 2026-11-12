import os
import sys
import time

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from stable_baselines3 import PPO


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from drl.mrm_environment import SafeStopEnv


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "mrm_ppo.zip"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Autonomous Safe-Stop MRM",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffffff;
        border: 1px solid #dddddd;
        text-align: center;
        min-height: 95px;
    }

    .status-title {
        font-size: 14px;
        color: #666666;
    }

    .status-value {
        font-size: 22px;
        font-weight: 700;
        margin-top: 8px;
    }

    .normal {
        color: #16803c;
    }

    .danger {
        color: #c62828;
    }

    .active {
        color: #1565c0;
    }

    .safe {
        color: #16803c;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">AUTONOMOUS SAFE-STOP MANEUVER SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'DRL-Based Minimum Risk Maneuver for Emergency Vehicle Pull-Over'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD PPO MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return PPO.load(MODEL_PATH)


model = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "simulation_started" not in st.session_state:
    st.session_state.simulation_started = False

if "simulation_finished" not in st.session_state:
    st.session_state.simulation_finished = False

if "final_result" not in st.session_state:
    st.session_state.final_result = "READY"

if "final_speed" not in st.session_state:
    st.session_state.final_speed = 20.0

if "shoulder_distance" not in st.session_state:
    st.session_state.shoulder_distance = 3.5

if "collision" not in st.session_state:
    st.session_state.collision = False

if "steps" not in st.session_state:
    st.session_state.steps = 0


# ============================================================
# TOP STATUS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        """
        <div class="status-box">
        <div class="status-title">DRIVER STATUS</div>
        <div class="status-value normal">● NORMAL</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="status-box">
        <div class="status-title">CONTROL AUTHORITY</div>
        <div class="status-value active">HUMAN</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="status-box">
        <div class="status-title">DRL POLICY</div>
        <div class="status-value active">PPO READY</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    if model is not None:

        model_status = "MODEL LOADED"

    else:

        model_status = "MODEL ERROR"

    st.markdown(
        f"""
        <div class="status-box">
        <div class="status-title">SYSTEM</div>
        <div class="status-value active">{model_status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# EMERGENCY BUTTON
# ============================================================

st.subheader("Emergency Control")

st.write(
    "Prototype control: manually trigger an emergency condition "
    "to transfer vehicle control from the driver to the MRM planner."
)


trigger = st.button(
    "🚨  INITIATE EMERGENCY SAFE-STOP",
    type="primary",
    use_container_width=True
)


# ============================================================
# MODEL ERROR
# ============================================================

if model is None:

    st.error(
        "PPO model could not be found. "
        "Expected: models/mrm_ppo.zip"
    )

    st.stop()


# ============================================================
# HIGHWAY VISUALIZATION
# ============================================================

st.subheader("Live Vehicle Maneuver")

chart_placeholder = st.empty()


# ============================================================
# METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

speed_placeholder = metric1.empty()
shoulder_placeholder = metric2.empty()
steering_placeholder = metric3.empty()
brake_placeholder = metric4.empty()


# ============================================================
# SYSTEM STATUS
# ============================================================

st.subheader("System Status")

status1, status2, status3, status4 = st.columns(4)

fusion_placeholder = status1.empty()
planner_placeholder = status2.empty()
control_placeholder = status3.empty()
result_placeholder = status4.empty()


# ============================================================
# DRAW HIGHWAY
# ============================================================

def draw_highway(
    vehicle_x,
    vehicle_y,
    obstacle_x,
    obstacle_y,
    safe_zone_y,
    title
):

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    # Road
    ax.fill_between(
        [0, 120],
        -2.5,
        2.5,
        alpha=0.18
    )

    # Shoulder
    ax.fill_between(
        [0, 120],
        -4.5,
        -2.5,
        alpha=0.12
    )

    # Lane markings
    ax.plot(
        [0, 120],
        [0, 0],
        linestyle="--",
        linewidth=1
    )

    # Shoulder boundary
    ax.plot(
        [0, 120],
        [-2.5, -2.5],
        linestyle="--",
        linewidth=2
    )

    # Vehicle
    ax.scatter(
        vehicle_x,
        vehicle_y,
        s=350,
        marker="s",
        label="EGO VEHICLE"
    )

    # Obstacle
    ax.scatter(
        obstacle_x,
        obstacle_y,
        s=350,
        marker="s",
        label="OBSTACLE"
    )

    # Safe stopping zone
    ax.axhspan(
        safe_zone_y - 0.5,
        safe_zone_y + 0.5,
        alpha=0.18
    )

    ax.text(
        vehicle_x,
        vehicle_y + 0.6,
        "EGO",
        ha="center",
        fontweight="bold"
    )

    ax.text(
        obstacle_x,
        obstacle_y + 0.6,
        "OBSTACLE",
        ha="center",
        fontweight="bold"
    )

    ax.text(
        95,
        -3.5,
        "SAFE FALLBACK ZONE",
        ha="center",
        fontweight="bold"
    )

    ax.set_xlim(
        max(0, vehicle_x - 20),
        min(120, vehicle_x + 60)
    )

    ax.set_ylim(
        -5,
        5
    )

    ax.set_xlabel(
        "Longitudinal Position (m)"
    )

    ax.set_ylabel(
        "Lateral Position (m)"
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold"
    )

    ax.legend(
        loc="upper right"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    return fig


# ============================================================
# INITIAL VIEW
# ============================================================

if not trigger:

    fig = draw_highway(
        vehicle_x=0,
        vehicle_y=0,
        obstacle_x=80,
        obstacle_y=0,
        safe_zone_y=-3.5,
        title="NORMAL DRIVING — WAITING FOR EMERGENCY TRIGGER"
    )

    chart_placeholder.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# EMERGENCY SIMULATION
# ============================================================

if trigger:

    st.session_state.simulation_started = True
    st.session_state.simulation_finished = False

    st.warning(
        "⚠ EMERGENCY CONDITION TRIGGERED — "
        "TRANSFERRING CONTROL TO AUTONOMOUS MRM"
    )

    # --------------------------------------------------------
    # INITIALIZE ENVIRONMENT
    # --------------------------------------------------------

    env = SafeStopEnv()

    observation, info = env.reset(
        seed=42
    )

    # Keep the demonstration obstacle deterministic
    env.obstacle_x = 80.0
    env.obstacle_y = 0.0

    # --------------------------------------------------------
    # RUN PPO
    # --------------------------------------------------------

    for step in range(100):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        steering = float(action[0])
        acceleration = float(action[1])

        # ----------------------------------------------------
        # APPROXIMATE BRAKING COMMAND FOR DISPLAY
        # ----------------------------------------------------

        brake_command = max(
            0.0,
            -acceleration
        )

        # ----------------------------------------------------
        # UPDATE METRICS
        # ----------------------------------------------------

        speed_placeholder.metric(
            "Vehicle Speed",
            f"{info['speed']:.2f} m/s"
        )

        shoulder_placeholder.metric(
            "Shoulder Distance",
            f"{info['shoulder_distance']:.2f} m"
        )

        steering_placeholder.metric(
            "Steering Command",
            f"{steering:.2f}"
        )

        brake_placeholder.metric(
            "Brake Command",
            f"{brake_command:.2f}"
        )

        # ----------------------------------------------------
        # SYSTEM STATUS
        # ----------------------------------------------------

        fusion_placeholder.success(
            "SENSOR FUSION\nSIMULATION"
        )

        planner_placeholder.success(
            "PPO MRM\nACTIVE"
        )

        control_placeholder.warning(
            "AI CONTROL\nACTIVE"
        )

        result_placeholder.info(
            "MANEUVER\nIN PROGRESS"
        )

        # ----------------------------------------------------
        # DRAW VEHICLE
        # ----------------------------------------------------

        fig = draw_highway(
            vehicle_x=env.x,
            vehicle_y=env.y,
            obstacle_x=env.obstacle_x,
            obstacle_y=env.obstacle_y,
            safe_zone_y=env.shoulder_y,
            title="AUTONOMOUS MINIMUM RISK MANEUVER — PPO ACTIVE"
        )

        chart_placeholder.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        time.sleep(0.08)

        if terminated or truncated:
            break

    # ========================================================
    # FINAL RESULT
    # ========================================================

    st.session_state.final_speed = info["speed"]
    st.session_state.shoulder_distance = info[
        "shoulder_distance"
    ]

    st.session_state.collision = info[
        "collision"
    ]

    st.session_state.steps = step + 1

    # --------------------------------------------------------
    # SAFE STOP
    # --------------------------------------------------------

    if info["safe_stop"] and not info["collision"]:

        st.session_state.final_result = "SAFE STOP"

        st.session_state.simulation_finished = True

        fusion_placeholder.success(
            "SENSOR FUSION\nCOMPLETE"
        )

        planner_placeholder.success(
            "PPO MRM\nSUCCESS"
        )

        control_placeholder.success(
            "CONTROL\nAUTONOMOUS"
        )

        result_placeholder.success(
            "✓ SAFE STOP"
        )

        st.success(
            "✓ VEHICLE HAS REACHED THE SAFE FALLBACK "
            "ZONE AND COMPLETED THE MINIMUM RISK MANEUVER."
        )

    elif info["collision"]:

        st.session_state.final_result = "COLLISION"

        result_placeholder.error(
            "COLLISION"
        )

        st.error(
            "Collision detected during simulation."
        )

    else:

        st.session_state.final_result = "TIMEOUT"

        result_placeholder.warning(
            "TIMEOUT"
        )

        st.warning(
            "Simulation ended before the safe-stop condition."
        )


# ============================================================
# FINAL INFORMATION
# ============================================================

st.write("")

st.subheader("Final Maneuver Summary")

summary1, summary2, summary3, summary4 = st.columns(4)

summary1.metric(
    "Final Speed",
    f"{st.session_state.final_speed:.2f} m/s"
)

summary2.metric(
    "Shoulder Distance",
    f"{st.session_state.shoulder_distance:.2f} m"
)

summary3.metric(
    "Collision",
    "YES" if st.session_state.collision else "NO"
)

summary4.metric(
    "Result",
    st.session_state.final_result
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

with st.expander(
    "Project Architecture"
):

    st.markdown(
        """
        ### Autonomous Safe-Stop Architecture

        **1. Manual Emergency Trigger**

        The driver manually activates the emergency condition
        for this prototype.

        **2. Control Authority Transfer**

        Human control is replaced by autonomous MRM control.

        **3. Sensor Information**

        Vehicle state, obstacle position and fallback-zone
        information are supplied to the planning environment.

        **4. DRL Planner**

        A trained PPO policy determines vehicle actions.

        **5. Minimum Risk Maneuver**

        The vehicle is guided toward the safe fallback zone
        while reducing speed.

        **6. Final Safe Stop**

        The maneuver terminates when the vehicle reaches the
        defined safe stopping zone at low speed.

        ---
        
        **Prototype limitation:** The current dashboard uses
        simulated sensor/environment information rather than
        physical Camera/LiDAR hardware.
        """
    )