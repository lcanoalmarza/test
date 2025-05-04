import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib.ticker import ScalarFormatter
import time

# Set page configuration
st.set_page_config(
    page_title="Bacterial Resistance Simulation",
    page_icon="🦠",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stSlider > div > div {
        color: #1E3A8A;
    }
    .stSidebar .sidebar-content {
        background-color: #F8F9FA;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("Bacterial Resistance Simulation")
st.markdown("""
This app simulates the spread of resistant bacteria in different environments.
Adjust the parameters using the sliders to see how they affect bacterial resistance over time.
""")

# Helper functions
def pnext(p, fitp, fitq, loss, conj):
    """Calculate the next proportion of resistant bacteria."""
    return (fitp * p - loss * p + p * (1 - p) * conj) / (fitp * p + fitq * (1 - p))

def simulate(i, g, l, loss, conj, fitq=1, timemax=60, sizex=50, xsteps=50):
    """Simulate the spread of resistant bacteria."""
    fitp = 1 + g - l
    tsteps = timemax * 72
    dt = timemax / tsteps
    dx = sizex / xsteps

    if dt > dx**2 / 2:
        st.warning('⚠️ Time step size too large for stability. Results may be inaccurate.')

    inicial = [0 for _ in range(xsteps + 2)]
    inicial[1 + xsteps // 2] = i

    etapas = [inicial]
    dibujo = [inicial[1:-1]]

    for _ in range(tsteps):
        siguiente = copy.deepcopy(etapas[-1])
        anterior = etapas[-1]
        for xx in range(1, xsteps + 1):
            espacialx = (1 / dx) ** 2 * (anterior[xx + 1] - 2 * anterior[xx] + anterior[xx - 1])
            temporal = (pnext(anterior[xx], fitp, fitq, loss, conj) - anterior[xx])/dt
            dP = temporal + espacialx
            siguiente[xx] = anterior[xx] + dt * dP
        etapas.append(siguiente)
        dibujo.append(siguiente[1:-1])

    return np.array(dibujo), fitp, tsteps, xsteps

# Create sidebar for parameters
st.sidebar.title("Simulation Parameters")
st.sidebar.markdown("Adjust the parameters below to see how they affect the simulation.")

# Create layout with two columns for parameters
col1, col2 = st.sidebar.columns(2)

# Presets dropdown
preset_options = {
    "Standard soil": {"i": 1e-9, "g": 0.005, "l": 0.001, "loss": 0.001, "conj": 1e-18},
    "Polluted soil": {"i": 1e-8, "g": 0.005, "l": 0.001, "loss": 0.001, "conj": 1e-16},
    "Custom": {}  # Will be populated by slider values
}

selected_preset = st.sidebar.selectbox(
    "Choose a preset or customize parameters:",
    options=list(preset_options.keys())
)

# Initialize session state for sliders if it doesn't exist
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.custom_params = preset_options["Standard soil"].copy()

# Update sliders based on preset selection (only when changed)
if selected_preset != "Custom" and ('last_preset' not in st.session_state or st.session_state.get('last_preset') != selected_preset):
    st.session_state.custom_params = preset_options[selected_preset].copy()
    st.session_state.last_preset = selected_preset

# Parameter sliders
with st.sidebar.expander("Basic Parameters", expanded=True):
    # Create sliders for parameters
    i_exp = st.slider("Initial proportion exponent (10^x)", 
                    min_value=-12, max_value=-6, 
                    value=int(np.log10(st.session_state.custom_params["i"])), 
                    step=1, 
                    help="Initial proportion of resistant bacteria (in scientific notation)")
    
    g = st.slider("Growth rate (g)", 
                min_value=0.001, max_value=0.01, 
                value=st.session_state.custom_params["g"], 
                step=0.001,
                format="%.3f",
                help="Growth rate of resistant bacteria")
    
    l = st.slider("Plasmid's fitness cost (λ)", 
               min_value=0.0001, max_value=0.005, 
               value=st.session_state.custom_params["l"], 
               step=0.0001,
               format="%.4f",
               help="Plasmid's fitness cost of resistant bacteria")

with st.sidebar.expander("Advanced Parameters", expanded=False):
    loss = st.slider("Loss rate", 
                   min_value=0.0001, max_value=0.005, 
                   value=st.session_state.custom_params["loss"], 
                   step=0.0001,
                   format="%.4f",
                   help="Rate at which resistant bacteria are lost from the system")
    
    conj_exp = st.slider("Conjugation rate exponent (10^x)", 
                       min_value=-20, max_value=-14, 
                       value=int(np.log10(st.session_state.custom_params["conj"])), 
                       step=1,
                       help="Rate of genetic transfer (conjugation) between bacteria (in scientific notation)")
    
    timemax = st.slider("Simulation duration (days)", 
                      min_value=1, max_value=120, 
                      value=60, 
                      step=10,
                      help="Total time to simulate in days")

# Convert slider exponent values to actual values
i = 10 ** i_exp
conj = 10 ** conj_exp

# Update custom params if in custom mode
if selected_preset == "Custom":
    st.session_state.custom_params = {"i": i, "g": g, "l": l, "loss": loss, "conj": conj}

# Add a "Run Simulation" button
run_sim = st.sidebar.button("Run Simulation", type="primary")

# Display current parameter values
st.sidebar.markdown("---")
st.sidebar.markdown("### Current Parameters")
params_display = f"""
- Initial proportion: {i:.2e}
- Growth rate (g): {g:.3f}
- Plasmid's fitness cost (λ): {l:.4f}
- Loss rate: {loss:.4f}
- Conjugation rate: {conj:.2e}
"""
st.sidebar.markdown(params_display)

# Add information about the model
with st.sidebar.expander("About the Model"):
    st.markdown("""
    This simulation models the spread of antibiotic resistance in bacterial populations.
    
    **Parameters explained:**
    - **Initial proportion**: The starting fraction of resistant bacteria
    - **Growth rate (g)**: How quickly resistant bacteria multiply
    - **Plasmid's fitness cost (λ)**: energy needed to maintain plasmid
    - **Loss rate**: Rate at which resistant bacteria are lost from the system
    - **Conjugation rate**: Rate of genetic transfer between bacteria
    
    The model combines spatial diffusion with temporal evolution to simulate how resistance spreads through space over time.
    """)

# Main content area
main_col1, main_col2 = st.columns([2, 1])

# Function to run the simulation and display results
def display_simulation(i, g, l, loss, conj, timemax):
    # Show a spinner while simulating
    with st.spinner("Running simulation..."):
        # Run simulation
        dibujo, fitp, tsteps, xsteps = simulate(i, g, l, loss, conj, timemax=timemax)
        
        # Create plot of time evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(range(tsteps + 1), [dibujo[i][xsteps // 2] for i in range(tsteps + 1)],
                label=f"IPP={i:.2e}, g={g:.3f}, λ={l:.4f}, loss={loss:.4f}, C={conj:.2e}",
                linewidth=3, color="#1E88E5")
        
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Resistant bacteria proportion', fontsize=12)
        ax.set_xticks(np.arange(0, tsteps + 1, 72 * 10))
        ax.set_xticklabels([i // 72 for i in range(0, tsteps + 1, 72 * 10)])
        
        # Move legend outside the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=10)
        ax.set_xlim(0, tsteps)
        
        # Nice numerical scale
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        return fig, dibujo, xsteps

# Function to create a heatmap visualization
def create_heatmap(dibujo, xsteps, timemax):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a time-space heatmap of the bacterial resistance
    extent = [0, xsteps, 0, timemax]
    
    # Downsample if needed for better display
    if dibujo.shape[0] > 200:
        step = dibujo.shape[0] // 200
        downsampled = dibujo[::step]
    else:
        downsampled = dibujo
    
    im = ax.imshow(downsampled, 
                  aspect='auto', 
                  origin='lower',
                  extent=extent,
                  cmap='viridis')
    
    plt.colorbar(im, ax=ax, label='Resistant bacteria proportion')
    
    ax.set_xlabel('Space', fontsize=12)
    ax.set_ylabel('Time (days)', fontsize=12)
    ax.set_title('Spatiotemporal Distribution of Resistant Bacteria', fontsize=14)
    
    plt.tight_layout()
    
    return fig

# Run the simulation when the button is clicked or when parameters change
if run_sim or 'last_params' not in st.session_state:
    # Store current parameters for comparison
    current_params = (i, g, l, loss, conj, timemax)
    
    # Only rerun if parameters have changed
    if 'last_params' not in st.session_state or st.session_state.last_params != current_params:
        with main_col1:
            st.subheader("Resistance over Time")
            fig, dibujo, xsteps = display_simulation(i, g, l, loss, conj, timemax)
            st.pyplot(fig)
            
            # Save results for heatmap
            st.session_state.dibujo = dibujo
            st.session_state.xsteps = xsteps
            st.session_state.timemax = timemax
        
        with main_col2:
            st.subheader("Spatial Distribution")
            heatmap_fig = create_heatmap(dibujo, xsteps, timemax)
            st.pyplot(heatmap_fig)
        
        # Metrics row
        metric_cols = st.columns(4)
        max_resistance = np.max([dibujo[i][xsteps // 2] for i in range(len(dibujo))])
        
        # Calculate when resistance reaches 50% of its maximum
        resistance_values = [dibujo[i][xsteps // 2] for i in range(len(dibujo))]
        half_max = max_resistance * 0.5
        time_to_half_max = next((i for i, r in enumerate(resistance_values) if r >= half_max), len(resistance_values))
        time_to_half_max_days = time_to_half_max / 72  # Convert to days
        
        # Final resistance value
        final_resistance = resistance_values[-1]
        
        # Calculate growth rate over initial period
        if len(resistance_values) > 100:
            initial_growth = (resistance_values[100] - resistance_values[0]) / (100 / 72)  # per day
        else:
            initial_growth = 0
            
        # Display metrics
        metric_cols[0].metric("Maximum Resistance", f"{max_resistance:.2e}")
        metric_cols[1].metric("Final Resistance", f"{final_resistance:.2e}")
        metric_cols[2].metric("Time to 50% Max (days)", f"{time_to_half_max_days:.1f}")
        metric_cols[3].metric("Initial Growth Rate", f"{initial_growth:.2e}/day")
        
        # Update last parameters
        st.session_state.last_params = current_params
    else:
        # Display cached results
        with main_col1:
            st.subheader("Resistance over Time")
            fig, dibujo, xsteps = display_simulation(i, g, l, loss, conj, timemax)
            st.pyplot(fig)
        
        with main_col2:
            st.subheader("Spatial Distribution")
            heatmap_fig = create_heatmap(st.session_state.dibujo, st.session_state.xsteps, st.session_state.timemax)
            st.pyplot(heatmap_fig)
            
        # Metrics row (same as above)
        metric_cols = st.columns(4)
        dibujo = st.session_state.dibujo
        xsteps = st.session_state.xsteps
        
        max_resistance = np.max([dibujo[i][xsteps // 2] for i in range(len(dibujo))])
        
        # Calculate when resistance reaches 50% of its maximum
        resistance_values = [dibujo[i][xsteps // 2] for i in range(len(dibujo))]
        half_max = max_resistance * 0.5
        time_to_half_max = next((i for i, r in enumerate(resistance_values) if r >= half_max), len(resistance_values))
        time_to_half_max_days = time_to_half_max / 72  # Convert to days
        
        # Final resistance value
        final_resistance = resistance_values[-1]
        
        # Calculate growth rate over initial period
        if len(resistance_values) > 100:
            initial_growth = (resistance_values[100] - resistance_values[0]) / (100 / 72)  # per day
        else:
            initial_growth = 0
            
        # Display metrics
        metric_cols[0].metric("Maximum Resistance", f"{max_resistance:.2e}")
        metric_cols[1].metric("Final Resistance", f"{final_resistance:.2e}")
        metric_cols[2].metric("Time to 50% Max (days)", f"{time_to_half_max_days:.1f}")
        metric_cols[3].metric("Initial Growth Rate", f"{initial_growth:.2e}/day")
else:
    # Show placeholder if simulation hasn't been run yet
    with main_col1:
        st.info("Click 'Run Simulation' to see results")
    with main_col2:
        st.info("Spatial distribution will appear here")

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    Bacterial Resistance Simulation Tool • Created with Streamlit
</div>
""", unsafe_allow_html=True)
