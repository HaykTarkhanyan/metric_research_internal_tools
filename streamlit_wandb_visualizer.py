import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import numpy as np
import os
from pathlib import Path
import wandb

def get_colors(num_runs):
    """Get colors for plots - Armenian flag colors if 3+ runs, otherwise default"""
    if num_runs >= 3:
        # Armenian flag colors: Red, Blue, Orange
        armenian_colors = ['#D90012', '#0033A0', '#F2A800']
        # Extend with additional colors if more than 3 runs
        if num_runs > 3:
            extra_colors = px.colors.qualitative.Set3[3:]
            return armenian_colors + extra_colors[:num_runs-3]
        return armenian_colors
    else:
        return px.colors.qualitative.Set3

# Page configuration
st.set_page_config(
    page_title="WandB Runs Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load and cache the wandb runs data"""
    try:
        df = pd.read_csv("wandb_runs_data.csv")
        return df
    except FileNotFoundError:
        st.error("wandb_runs_data.csv not found. Please make sure the file is in the same directory.")
        return None

def parse_history(history_str):
    """Parse history string into DataFrame"""
    try:
        if isinstance(history_str, str):
            # Replace 'nan' with 'None' to make it parseable
            cleaned_str = history_str.replace('nan', 'None')
            history_list = ast.literal_eval(cleaned_str)
        else:
            history_list = history_str
        
        if history_list and len(history_list) > 0:
            df_hist = pd.DataFrame(history_list)
            # Convert None back to NaN
            df_hist = df_hist.replace({None: np.nan})
            return df_hist
        else:
            return pd.DataFrame()
    except Exception:
        # If ast.literal_eval fails, try eval as a last resort (be careful with this!)
        try:
            if isinstance(history_str, str):
                # Create a safe environment for eval
                safe_dict = {"nan": float('nan'), "__builtins__": {}}
                history_list = eval(history_str, safe_dict)
                if history_list and len(history_list) > 0:
                    return pd.DataFrame(history_list)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

def get_available_metrics(df):
    """Get all available metrics from all runs"""
    all_metrics = set()
    
    for idx, row in df.iterrows():
        hist_df = parse_history(row['history'])
        if not hist_df.empty:
            # Get numeric columns (exclude timestamp and step columns for metric selection)
            numeric_cols = hist_df.select_dtypes(include=[np.number]).columns.tolist()
            all_metrics.update(numeric_cols)
    
    # Remove common non-metric columns
    exclude_cols = {'_timestamp', '_runtime', '_step'}
    all_metrics = all_metrics - exclude_cols
    
    return sorted(list(all_metrics))

def create_comparison_plot(df, selected_runs, metric, plot_type="line", log_x=False):
    """Create comparison plot for selected runs and metric"""
    fig = go.Figure()
    
    colors = get_colors(len(selected_runs))
    
    for i, run_name in enumerate(selected_runs):
        run_data = df[df['name'] == run_name].iloc[0]
        hist_df = parse_history(run_data['history'])
        
        if not hist_df.empty and metric in hist_df.columns:
            # Use global_step if available, otherwise use index
            x_col = None
            for step_col in ['train/global_step', 'step', '_step']:
                if step_col in hist_df.columns:
                    x_col = step_col
                    break
            
            x_data = hist_df[x_col] if x_col else hist_df.index
            y_data = hist_df[metric].dropna()
            x_data = x_data[:len(y_data)]  # Match lengths
            
            color = colors[i % len(colors)]
            
            if plot_type == "line":
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=run_name,
                    line=dict(color=color)
                ))
            elif plot_type == "scatter":
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    name=run_name,
                    marker=dict(color=color)
                ))
    
    x_title = "Step (log scale)" if log_x else "Step"
    
    fig.update_layout(
        title=f"{metric} Comparison",
        xaxis_title=x_title,
        yaxis_title=metric,
        height=600,
        hovermode='x unified'
    )
    
    if log_x:
        fig.update_xaxes(type="log")
    
    return fig

def create_individual_plots(df, selected_runs, metric, log_x=False):
    """Create individual subplots for each run"""
    n_runs = len(selected_runs)
    
    # Handle single run case
    if n_runs == 1:
        cols = 1
        rows = 1
    else:
        cols = min(2, n_runs)
        rows = (n_runs + cols - 1) // cols
    
    # Create subplots layout
    
    try:
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=selected_runs,
            vertical_spacing=0.15 if rows > 1 else 0.1,
            horizontal_spacing=0.1
        )
        pass
    except Exception:
        # Fallback to simple figure
        fig = go.Figure()
        rows = 1
        cols = 1
    
    colors = get_colors(len(selected_runs))
    
    for i, run_name in enumerate(selected_runs):
        if rows > 1 or cols > 1:
            row = i // cols + 1
            col = i % cols + 1
        else:
            row = 1
            col = 1
        
        color = colors[i % len(colors)]
        
        # Process each run
        
        try:
            run_data = df[df['name'] == run_name].iloc[0]
            hist_df = parse_history(run_data['history'])
            
            if not hist_df.empty and metric in hist_df.columns:
                # Use global_step if available, otherwise use index
                x_col = None
                for step_col in ['train/global_step', 'step', '_step']:
                    if step_col in hist_df.columns:
                        x_col = step_col
                        break
                
                x_data = hist_df[x_col] if x_col else hist_df.index
                y_data = hist_df[metric].dropna()
                x_data = x_data[:len(y_data)]  # Match lengths
                
                # Add trace to plot
                
                if rows == 1 and cols == 1:
                    # Single plot case
                    fig.add_trace(
                        go.Scatter(x=x_data, y=y_data, mode='lines', name=run_name, 
                                 line=dict(color=color), showlegend=True)
                    )
                else:
                    # Multiple subplots case
                    fig.add_trace(
                        go.Scatter(x=x_data, y=y_data, mode='lines', name=run_name, 
                                 line=dict(color=color), showlegend=False),
                        row=row, col=col
                    )
            else:
                pass  # No data for this run
        except Exception:
            pass  # Error processing this run
    
    x_title = "Step (log scale)" if log_x else "Step"
    
    try:
        fig.update_layout(
            title=f"{metric} - Individual Plots",
            height=max(400, 300 * rows),  # Minimum height of 400
        )
        
        if log_x:
            fig.update_xaxes(type="log")
        
        # Update all x-axis labels
        for i in range(1, rows + 1):
            for j in range(1, min(2, n_runs) + 1):
                fig.update_xaxes(title_text=x_title, row=i, col=j)
                fig.update_yaxes(title_text=metric, row=i, col=j)
        
        pass
    except Exception:
        pass
    
    return fig

def create_trimmed_comparison_plot(df, selected_runs, metric, plot_type="line", percentage=None, fixed_rows=None, log_x=False):
    """Create comparison plot for selected runs and metric with trimming"""
    fig = go.Figure()
    
    colors = get_colors(len(selected_runs))
    
    for i, run_name in enumerate(selected_runs):
        run_data = df[df['name'] == run_name].iloc[0]
        hist_df = parse_history(run_data['history'])
        
        if not hist_df.empty and metric in hist_df.columns:
            # Use global_step if available, otherwise use index
            x_col = None
            for step_col in ['train/global_step', 'step', '_step']:
                if step_col in hist_df.columns:
                    x_col = step_col
                    break
            
            x_data = hist_df[x_col] if x_col else hist_df.index
            y_data = hist_df[metric].dropna()
            x_data = x_data[:len(y_data)]  # Match lengths
            
            # Apply trimming
            if percentage is not None:
                skip_count = int(len(y_data) * percentage / 100)
            elif fixed_rows is not None:
                skip_count = min(fixed_rows, len(y_data) - 1)
            else:
                skip_count = 0
            
            # Trim the data
            if skip_count > 0 and skip_count < len(y_data):
                y_data_trimmed = y_data.iloc[skip_count:]
                # Trim x_data to match - use iloc for pandas Series or simple slicing for arrays
                if hasattr(x_data, 'iloc'):
                    x_data_trimmed = x_data.iloc[skip_count:]
                else:
                    x_data_trimmed = x_data[skip_count:]
            else:
                y_data_trimmed = y_data
                x_data_trimmed = x_data
            
            color = colors[i % len(colors)]
            
            if plot_type == "line":
                fig.add_trace(go.Scatter(
                    x=x_data_trimmed,
                    y=y_data_trimmed,
                    mode='lines',
                    name=f"{run_name} (trimmed)",
                    line=dict(color=color)
                ))
            elif plot_type == "scatter":
                fig.add_trace(go.Scatter(
                    x=x_data_trimmed,
                    y=y_data_trimmed,
                    mode='markers',
                    name=f"{run_name} (trimmed)",
                    marker=dict(color=color)
                ))
    
    trim_info = ""
    if percentage is not None:
        trim_info = f" (First {percentage}% Trimmed)"
    elif fixed_rows is not None:
        trim_info = f" (First {fixed_rows} Steps Trimmed)"
    
    x_title = "Step (log scale)" if log_x else "Step"
    
    fig.update_layout(
        title=f"{metric} Comparison{trim_info}",
        xaxis_title=x_title,
        yaxis_title=metric,
        height=600,
        hovermode='x unified'
    )
    
    if log_x:
        fig.update_xaxes(type="log")
    
    return fig

def create_trimmed_individual_plots(df, selected_runs, metric, percentage=None, fixed_rows=None, log_x=False):
    """Create individual subplots for each run with trimming"""
    n_runs = len(selected_runs)
    cols = min(2, n_runs)
    rows = (n_runs + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[f"{run} (trimmed)" for run in selected_runs],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    colors = get_colors(len(selected_runs))
    
    for i, run_name in enumerate(selected_runs):
        row = i // cols + 1
        col = i % cols + 1
        color = colors[i % len(colors)]
        
        run_data = df[df['name'] == run_name].iloc[0]
        hist_df = parse_history(run_data['history'])
        
        if not hist_df.empty and metric in hist_df.columns:
            # Use global_step if available, otherwise use index
            x_col = None
            for step_col in ['train/global_step', 'step', '_step']:
                if step_col in hist_df.columns:
                    x_col = step_col
                    break
            
            x_data = hist_df[x_col] if x_col else hist_df.index
            y_data = hist_df[metric].dropna()
            x_data = x_data[:len(y_data)]  # Match lengths
            
            # Apply trimming
            if percentage is not None:
                skip_count = int(len(y_data) * percentage / 100)
            elif fixed_rows is not None:
                skip_count = min(fixed_rows, len(y_data) - 1)
            else:
                skip_count = 0
            
            # Trim the data
            if skip_count > 0 and skip_count < len(y_data):
                y_data_trimmed = y_data.iloc[skip_count:]
                # Trim x_data to match - use iloc for pandas Series or simple slicing for arrays
                if hasattr(x_data, 'iloc'):
                    x_data_trimmed = x_data.iloc[skip_count:]
                else:
                    x_data_trimmed = x_data[skip_count:]
            else:
                y_data_trimmed = y_data
                x_data_trimmed = x_data
            
            fig.add_trace(
                go.Scatter(x=x_data_trimmed, y=y_data_trimmed, mode='lines', 
                          name=run_name, line=dict(color=color), showlegend=False),
                row=row, col=col
            )
    
    trim_info = ""
    if percentage is not None:
        trim_info = f" (First {percentage}% Trimmed)"
    elif fixed_rows is not None:
        trim_info = f" (First {fixed_rows} Steps Trimmed)"
    
    x_title = "Step (log scale)" if log_x else "Step"
    
    fig.update_layout(
        title=f"{metric} - Individual Plots{trim_info}",
        height=300 * rows,
    )
    
    if log_x:
        fig.update_xaxes(type="log")
    
    # Update all x-axis labels
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(title_text=x_title, row=i, col=j)
            fig.update_yaxes(title_text=metric, row=i, col=j)
    
    return fig

def detect_spikes_and_dips_local(data, spike_threshold=10.0, dip_threshold=10.0, min_distance=5, 
                                prominence_threshold=0.1, smoothing_window=2, local_window=5):
    """Detect spikes and dips using percentage comparison with neighboring points"""
    try:
        import numpy as np
        
        # Convert to numpy array
        values = np.array(data)
        
        if len(values) < 7:  # Need at least 7 points for 5 neighbors + center + 1 buffer
            return {
                'spike_indices': np.array([]),
                'spike_values': np.array([]),
                'dip_indices': np.array([]),
                'dip_values': np.array([]),
                'smoothed_data': values,
                'local_stats': []
            }
        
        # Apply smoothing (default window=2)
        if smoothing_window > 1:
            smoothed = np.convolve(values, np.ones(smoothing_window)/smoothing_window, mode='same')
        else:
            smoothed = values
        
        # Find spikes and dips using percentage comparison
        spike_candidates = []
        dip_candidates = []
        local_stats = []
        
        # Use local_window as the number of neighboring points to compare (default=5)
        neighbors = local_window
        
        for i in range(neighbors, len(smoothed) - neighbors):
            # Get neighboring points (exclude the center point itself)
            left_neighbors = smoothed[i-neighbors:i]
            right_neighbors = smoothed[i+1:i+neighbors+1]
            all_neighbors = np.concatenate([left_neighbors, right_neighbors])
            
            # Calculate average of neighboring points
            neighbor_avg = np.mean(all_neighbors)
            current_value = smoothed[i]
            
            # Store local stats for debugging
            local_stats.append({
                'index': i,
                'current_value': current_value,
                'neighbor_avg': neighbor_avg,
                'neighbors': all_neighbors.tolist()
            })
            
            # Check for spikes: current value is spike_threshold% higher than neighbor average
            if abs(neighbor_avg) > 1e-10:  # Avoid division by zero with very small threshold
                percent_increase = ((current_value - neighbor_avg) / abs(neighbor_avg)) * 100
                if percent_increase >= spike_threshold:
                    spike_candidates.append(i)
            elif current_value > neighbor_avg:  # Handle zero/near-zero case
                spike_candidates.append(i)
            
            # Check for dips: current value is dip_threshold% lower than neighbor average  
            if abs(neighbor_avg) > 1e-10:  # Avoid division by zero with very small threshold
                percent_decrease = ((neighbor_avg - current_value) / abs(neighbor_avg)) * 100
                if percent_decrease >= dip_threshold:
                    dip_candidates.append(i)
            elif current_value < neighbor_avg:  # Handle zero/near-zero case
                dip_candidates.append(i)
        
        # Simple distance filtering - keep all candidates but enforce minimum distance
        spike_indices = []
        if spike_candidates:
            spike_candidates = sorted(spike_candidates)
            spike_indices.append(spike_candidates[0])
            for candidate in spike_candidates[1:]:
                if candidate - spike_indices[-1] >= min_distance:
                    spike_indices.append(candidate)
        
        dip_indices = []
        if dip_candidates:
            dip_candidates = sorted(dip_candidates)
            dip_indices.append(dip_candidates[0])
            for candidate in dip_candidates[1:]:
                if candidate - dip_indices[-1] >= min_distance:
                    dip_indices.append(candidate)
        
        return {
            'spike_indices': np.array(spike_indices),
            'spike_values': values[spike_indices] if spike_indices else np.array([]),
            'dip_indices': np.array(dip_indices),
            'dip_values': values[dip_indices] if dip_indices else np.array([]),
            'smoothed_data': smoothed,
            'local_stats': local_stats
        }
        
    except Exception as e:
        st.error(f"Error in anomaly detection: {str(e)}")
        return {
            'spike_indices': np.array([]),
            'spike_values': np.array([]),
            'dip_indices': np.array([]),
            'dip_values': np.array([]),
            'smoothed_data': np.array(data),
            'local_stats': []
        }





def create_individual_anomaly_plots(df, selected_runs, metric, spike_threshold, 
                                   dip_threshold, min_distance, prominence_threshold, 
                                   smoothing_window, local_window, log_x=False):
    """Create individual plots for each run with local anomaly detection"""
    from plotly.subplots import make_subplots
    
    num_runs = len(selected_runs)
    cols = 2
    rows = (num_runs + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=selected_runs,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    colors = get_colors(len(selected_runs))
    all_anomalies = []
    
    for i, run_name in enumerate(selected_runs):
        row = i // cols + 1
        col = i % cols + 1
        
        run_data = df[df['name'] == run_name].iloc[0]
        hist_df = parse_history(run_data['history'])
        
        if not hist_df.empty and metric in hist_df.columns:
            # Get x and y data
            x_col = None
            for step_col in ['train/global_step', 'step', '_step']:
                if step_col in hist_df.columns:
                    x_col = step_col
                    break
            
            x_data = hist_df[x_col] if x_col else hist_df.index
            y_data = hist_df[metric].dropna()
            x_data = x_data[:len(y_data)]
            
            if len(y_data) == 0:
                continue
                
            color = colors[i % len(colors)]
            
            # Detect anomalies using local statistics
            anomaly_results = detect_spikes_and_dips_local(
                y_data, spike_threshold, dip_threshold, min_distance, 
                prominence_threshold, smoothing_window, local_window
            )
            
            # Plot original data
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                name=f"{run_name}",
                line=dict(color=color, width=2),
                opacity=0.7,
                showlegend=False
            ), row=row, col=col)
            
            # Plot smoothed data if different
            if smoothing_window > 1:
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=anomaly_results['smoothed_data'],
                    mode='lines',
                    name=f"{run_name} (smoothed)",
                    line=dict(color=color, width=1, dash='dash'),
                    opacity=0.5,
                    showlegend=False
                ), row=row, col=col)
            
            # Add spikes
            if len(anomaly_results['spike_indices']) > 0:
                spike_x = x_data[anomaly_results['spike_indices']]
                spike_y = anomaly_results['spike_values']
                
                fig.add_trace(go.Scatter(
                    x=spike_x,
                    y=spike_y,
                    mode='markers',
                    name="Spikes" if i == 0 else None,
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='triangle-up',
                        line=dict(color='darkred', width=1)
                    ),
                    showlegend=(i == 0)
                ), row=row, col=col)
                
                # Store anomaly info
                for x_val, y_val in zip(spike_x, spike_y):
                    all_anomalies.append({
                        'run': run_name,
                        'type': 'Spike',
                        'step': x_val,
                        'value': y_val
                    })
            
            # Add dips
            if len(anomaly_results['dip_indices']) > 0:
                dip_x = x_data[anomaly_results['dip_indices']]
                dip_y = anomaly_results['dip_values']
                
                fig.add_trace(go.Scatter(
                    x=dip_x,
                    y=dip_y,
                    mode='markers',
                    name="Dips" if i == 0 else None,
                    marker=dict(
                        color='blue',
                        size=10,
                        symbol='triangle-down',
                        line=dict(color='darkblue', width=1)
                    ),
                    showlegend=(i == 0)
                ), row=row, col=col)
                
                # Store anomaly info
                for x_val, y_val in zip(dip_x, dip_y):
                    all_anomalies.append({
                        'run': run_name,
                        'type': 'Dip',
                        'step': x_val,
                        'value': y_val
                    })
    
    x_title = "Step (log scale)" if log_x else "Step"
    
    fig.update_layout(
        title=f"{metric} - Individual Anomaly Detection (Local Statistics)",
        height=300 * rows,
        showlegend=True
    )
    
    if log_x:
        fig.update_xaxes(type="log")
    
    # Update all x-axis labels
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(title_text=x_title, row=i, col=j)
            fig.update_yaxes(title_text=metric, row=i, col=j)
    
    return fig, all_anomalies

def show_anomaly_detection_view(df, selected_runs, selected_metric, plot_type, view_mode,
                               spike_threshold, dip_threshold, min_distance, 
                               prominence_threshold, smoothing_window, local_window, sensitivity_preset, log_x=False):
    """Show the anomaly detection view with spike and dip detection - Individual plots only"""
    
    # Always use individual plots with local statistics
    fig, anomalies = create_individual_anomaly_plots(
        df, selected_runs, selected_metric,
        spike_threshold, dip_threshold, min_distance, 
        prominence_threshold, smoothing_window, local_window, log_x
    )
    
    st.subheader(f"🔍 {selected_metric} - Individual Anomaly Detection (Local Statistics)")
    st.plotly_chart(fig, use_container_width=True)
        
    # Show anomaly summary
    if anomalies:
        st.subheader("📊 Detected Anomalies")
        anomaly_df = pd.DataFrame(anomalies)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_spikes = len(anomaly_df[anomaly_df['type'] == 'Spike'])
            st.metric("Total Spikes", total_spikes)
        with col2:
            total_dips = len(anomaly_df[anomaly_df['type'] == 'Dip'])
            st.metric("Total Dips", total_dips)
        with col3:
            total_anomalies = len(anomaly_df)
            st.metric("Total Anomalies", total_anomalies)
        
        # Detailed table
        st.dataframe(
            anomaly_df.round(4),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No anomalies detected with current parameters. Try adjusting the thresholds.")
    
    # Detection parameters summary
    with st.expander("🔧 Detection Parameters"):
        st.write("**Statistics Mode:** Local (mean/std calculated from local windows)")
        st.write(f"**Sensitivity Preset:** {sensitivity_preset}")
        st.write(f"**Local Window Size:** {local_window} steps")
        st.write(f"**Spike Threshold:** {spike_threshold} standard deviations above local mean")
        st.write(f"**Dip Threshold:** {dip_threshold} standard deviations below local mean")
        st.write(f"**Minimum Distance:** {min_distance} steps between peaks")
        st.write(f"**Prominence Threshold:** {prominence_threshold}")
        st.write(f"**Smoothing Window:** {smoothing_window} steps")
        if smoothing_window > 1:
            st.write("*Data is smoothed using moving average before detection*")
        st.write("💡 *Only individual plots are shown for better local context*")

def load_api_key_from_env():
    """Load WandB API key from .env file"""
    try:
        # Look for .env file in current directory and parent directories
        current_dir = Path.cwd()
        env_paths = [
            current_dir / ".env",
            current_dir.parent / ".env",
            current_dir.parent.parent / ".env"
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith('WANDB_API_KEY='):
                            return line.strip().split('=', 1)[1]
        return None
    except Exception as e:
        st.error(f"Error loading API key from .env: {e}")
        return None

def authenticate_wandb(api_key):
    """Authenticate with WandB using the provided API key"""
    try:
        if wandb is None:
            st.error("WandB is not installed. Please install it with: pip install wandb")
            return False
        
        os.environ['WANDB_API_KEY'] = api_key
        wandb.login(key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to authenticate with WandB: {e}")
        return False

def main():
    st.set_page_config(page_title="W&B Runs Visualizer", layout="wide")
    
    st.title("📊 WandB Runs Visualizer")
    st.markdown("Visualize and compare metrics across multiple WandB runs")
    
    # Sidebar for WandB authentication
    st.sidebar.header("🔐 WandB Authentication")
    
    # Check if already authenticated
    if 'wandb_authenticated' not in st.session_state:
        st.session_state.wandb_authenticated = False
    
    if not st.session_state.wandb_authenticated:
        st.sidebar.markdown("Enter your WandB API key to access your runs:")
        api_key_input = st.sidebar.text_input(
            "API Key", 
            type="password", 
            placeholder="Enter API key or 'panir' to load from .env",
            help="Enter 'panir' to automatically load the API key from .env file"
        )
        
        if st.sidebar.button("🚀 Login to WandB"):
            if api_key_input:
                if api_key_input.lower() == "panir":
                    # Load from .env
                    env_api_key = load_api_key_from_env()
                    if env_api_key:
                        if authenticate_wandb(env_api_key):
                            st.session_state.wandb_authenticated = True
                            st.sidebar.success("✅ Successfully authenticated with WandB using .env key!")
                            st.rerun()
                        else:
                            st.sidebar.error("❌ Failed to authenticate with WandB")
                    else:
                        st.sidebar.error("❌ Could not find WANDB_API_KEY in .env file")
                else:
                    # Use provided API key
                    if authenticate_wandb(api_key_input):
                        st.session_state.wandb_authenticated = True
                        st.sidebar.success("✅ Successfully authenticated with WandB!")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ Failed to authenticate with WandB")
            else:
                st.sidebar.error("❌ Please enter an API key")
    else:
        st.sidebar.success("✅ WandB Authenticated")
        if st.sidebar.button("🚪 Logout"):
            st.session_state.wandb_authenticated = False
            if 'WANDB_API_KEY' in os.environ:
                del os.environ['WANDB_API_KEY']
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Only show the rest of the interface if authenticated
    if not st.session_state.wandb_authenticated:
        st.info("🔐 Please authenticate with WandB using the sidebar to access your runs.")
        st.markdown("""
        ### How to get your WandB API key:
        1. Go to [WandB Settings](https://wandb.ai/authorize)
        2. Copy your API key
        3. Paste it in the sidebar and click "Login to WandB"
        
        **Special shortcut:** Type `panir` to automatically load the API key from your .env file.
        """)
        return
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Global sidebar controls
    st.sidebar.header("🎛️ Global Controls")
    
    # Run selection
    st.sidebar.subheader("Select Runs")
    available_runs = df['name'].tolist()
    
    select_all = st.sidebar.checkbox("Select All Runs")
    if select_all:
        selected_runs = st.sidebar.multiselect(
            "Choose runs to visualize:",
            available_runs,
            default=available_runs
        )
    else:
        selected_runs = st.sidebar.multiselect(
            "Choose runs to visualize:",
            available_runs,
            default=available_runs[:3] if len(available_runs) >= 3 else available_runs
        )
    
    if not selected_runs:
        st.warning("Please select at least one run to visualize.")
        return
    
    # Metric selection
    st.sidebar.subheader("Select Metric")
    available_metrics = get_available_metrics(df[df['name'].isin(selected_runs)])
    
    if not available_metrics:
        st.error("No metrics found in selected runs.")
        return
    
    # Set default metric
    default_metric = None
    for common_metric in ['train/loss', 'eval/loss', 'loss']:
        if common_metric in available_metrics:
            default_metric = common_metric
            break
    
    selected_metric = st.sidebar.selectbox(
        "Choose metric to visualize:",
        available_metrics,
        index=available_metrics.index(default_metric) if default_metric else 0
    )
    
    # Plot options
    st.sidebar.subheader("Plot Options")
    plot_type = st.sidebar.radio("Plot Type:", ["line", "scatter"])
    view_mode = st.sidebar.radio("View Mode:", ["Comparison", "Individual"])
    log_x = st.sidebar.checkbox("Logarithmic X-axis", value=False, help="Transform x-axis to logarithmic scale")
    
    # Trimming options (for trimmed view)
    st.sidebar.subheader("✂️ Trimming Options")
    
    # Calculate average data length for percentage calculation
    avg_length = 0
    count = 0
    for run_name in selected_runs:
        run_data = df[df['name'] == run_name].iloc[0]
        hist_df = parse_history(run_data['history'])
        if not hist_df.empty and selected_metric in hist_df.columns:
            values = hist_df[selected_metric].dropna()
            if not values.empty:
                avg_length += len(values)
                count += 1
    
    if count > 0:
        avg_length = avg_length // count
        default_skip_rows = max(1, int(avg_length * 0.1))  # 10% by default
    else:
        default_skip_rows = 10
    
    trim_mode = st.sidebar.radio(
        "Trimming Mode:",
        ["Percentage", "Fixed Number"]
    )
    
    # Initialize variables
    skip_percentage = None
    skip_rows = None
    
    if trim_mode == "Percentage":
        skip_percentage = st.sidebar.slider(
            "Skip first X% of data points:",
            min_value=0,
            max_value=100,
            value=10,
            step=1
        )
        # Show individual step counts for each run
        step_info = []
        for run_name in selected_runs:
            run_data = df[df['name'] == run_name].iloc[0]
            hist_df = parse_history(run_data['history'])
            if not hist_df.empty and selected_metric in hist_df.columns:
                values = hist_df[selected_metric].dropna()
                if not values.empty:
                    skip_count = int(len(values) * skip_percentage / 100)
                    step_info.append(f"{run_name}: {skip_count} steps")
        
        if step_info:
            st.sidebar.info("Steps to skip per run:\n" + "\n".join(step_info))
    else:
        skip_rows = st.sidebar.number_input(
            "Skip first X steps:",
            min_value=0,
            max_value=1000,
            value=default_skip_rows,
            step=1
        )
    
    # Spike/Dip Detection options (for anomaly detection view)
    st.sidebar.subheader("🔍 Spike/Dip Detection")
    
    # Preset sensitivity levels
    sensitivity_preset = st.sidebar.selectbox(
        "Detection Sensitivity:",
        ["Very Strong Peaks/Dips Only", "Moderate Peaks/Dips", "Custom"],
        index=0,
        help="Choose preset parameters or customize manually"
    )
    
    # Set parameters based on preset
    if sensitivity_preset == "Very Strong Peaks/Dips Only":
        spike_threshold = 8.0
        dip_threshold = 8.0
        min_distance = 2
        prominence_threshold = 0.05
        smoothing_window = 2
        local_window = 3
        st.sidebar.info("🎯 **Very Strong**: Detects 8%+ changes vs neighbors")
    
    elif sensitivity_preset == "Moderate Peaks/Dips":
        spike_threshold = 5.0
        dip_threshold = 5.0
        min_distance = 1
        prominence_threshold = 0.02
        smoothing_window = 2
        local_window = 3
        st.sidebar.info("📊 **Moderate**: Detects 5%+ changes vs neighbors")
    
    else:  # Custom
        # Peak detection parameters
        spike_threshold = st.sidebar.slider(
            "Spike Threshold (% above neighbors):",
            min_value=1.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            help="Spikes must be this % higher than average of neighboring points"
        )
        
        dip_threshold = st.sidebar.slider(
            "Dip Threshold (% below neighbors):",
            min_value=1.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            help="Dips must be this % lower than average of neighboring points"
        )
        
        min_distance = st.sidebar.number_input(
            "Minimum Distance Between Peaks:",
            min_value=1,
            max_value=100,
            value=1,
            help="Minimum number of steps between detected spikes/dips"
        )
        
        prominence_threshold = st.sidebar.slider(
            "Prominence Threshold:",
            min_value=0.01,
            max_value=2.0,
            value=0.02,
            step=0.01,
            help="How prominent peaks/dips must be relative to their surroundings"
        )
        
        smoothing_window = st.sidebar.number_input(
            "Smoothing Window Size:",
            min_value=1,
            max_value=50,
            value=2,
            help="Size of moving average window for smoothing (default = 2)"
        )
        
        local_window = st.sidebar.number_input(
            "Number of Neighbors:",
            min_value=3,
            max_value=20,
            value=3,
            help="Number of neighboring points to compare with on each side"
        )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Standard View", "✂️ Trimmed View", "🔍 Anomaly Detection"])
    
    with tab1:
        show_standard_view(df, selected_runs, selected_metric, plot_type, view_mode, log_x)
    
    with tab2:
        show_trimmed_view(df, selected_runs, selected_metric, plot_type, view_mode, 
                         trim_mode, skip_percentage, skip_rows, log_x)
    
    with tab3:
        show_anomaly_detection_view(df, selected_runs, selected_metric, plot_type, view_mode,
                                   spike_threshold, dip_threshold, min_distance, 
                                   prominence_threshold, smoothing_window, local_window, sensitivity_preset, log_x)

def show_standard_view(df, selected_runs, selected_metric, plot_type, view_mode, log_x=False):
    """Show the standard view with all data points"""
    
    # Main content
    # col1, col2 = st.columns([3, 1])
    
    # with col1:
    st.subheader(f"📈 {selected_metric}")
    
    if view_mode == "Comparison":
        fig = create_comparison_plot(df, selected_runs, selected_metric, plot_type, log_x)
    else:
        fig = create_individual_plots(df, selected_runs, selected_metric, log_x)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # with col2:
    #     st.subheader("📋 Run Info")
    #     
    #     # Show basic stats
    #     st.markdown("**Selected Runs:**")
    #     for run_name in selected_runs:
    #         run_data = df[df['name'] == run_name].iloc[0]
    #         st.markdown(f"• {run_name}")
    #         st.markdown(f"  *State: {run_data['state']}*")
    #     
    #     # Show metric statistics
    #     st.markdown(f"**{selected_metric} Stats:**")
    #     metric_stats = []
    #     
    #     for run_name in selected_runs:
    #         run_data = df[df['name'] == run_name].iloc[0]
    #         hist_df = parse_history(run_data['history'])
    #         
    #         if not hist_df.empty and selected_metric in hist_df.columns:
    #             values = hist_df[selected_metric].dropna()
    #             if not values.empty:
    #                 stats = {
    #                     'Run': run_name,
    #                     'Min': f"{values.min():.4f}",
    #                     'Max': f"{values.max():.4f}",
    #                     'Final': f"{values.iloc[-1]:.4f}",
    #                     'Points': len(values)
    #                 }
    #                 metric_stats.append(stats)
    #     
    #     if metric_stats:
    #         stats_df = pd.DataFrame(metric_stats)
    #         st.dataframe(stats_df, hide_index=True)
    
    # Additional information
    available_metrics = get_available_metrics(df[df['name'].isin(selected_runs)])
    with st.expander("📊 Dataset Information"):
        st.write(f"**Total Runs:** {len(df)}")
        st.write(f"**Selected Runs:** {len(selected_runs)}")
        st.write(f"**Available Metrics:** {len(available_metrics)}")
        st.write("**All Available Metrics:**")
        st.write(", ".join(available_metrics))

def show_trimmed_view(df, selected_runs, selected_metric, plot_type, view_mode, trim_mode, skip_percentage, skip_rows, log_x=False):
    """Show the trimmed view with ability to discard first x rows"""
    
    # Main content
    # col1, col2 = st.columns([3, 1])
    
    # with col1:
    # Create a unique subtitle that changes with parameters to force refresh
    if trim_mode == "Percentage" and skip_percentage is not None:
        subtitle = f"✂️ {selected_metric} (First {skip_percentage}% Trimmed)"
    elif trim_mode == "Fixed Number" and skip_rows is not None:
        subtitle = f"✂️ {selected_metric} (First {skip_rows} Steps Trimmed)"
    else:
        subtitle = f"✂️ {selected_metric} (Trimmed)"
        
    st.subheader(subtitle)
    
    # Create plot with current parameters
    if view_mode == "Comparison":
        if trim_mode == "Percentage" and skip_percentage is not None:
            fig = create_trimmed_comparison_plot(df, selected_runs, selected_metric, plot_type, 
                                               percentage=skip_percentage, log_x=log_x)
        elif trim_mode == "Fixed Number" and skip_rows is not None:
            fig = create_trimmed_comparison_plot(df, selected_runs, selected_metric, plot_type, 
                                               fixed_rows=skip_rows, log_x=log_x)
        else:
            # Fallback to no trimming
            fig = create_comparison_plot(df, selected_runs, selected_metric, plot_type, log_x)
    else:
        if trim_mode == "Percentage" and skip_percentage is not None:
            fig = create_trimmed_individual_plots(df, selected_runs, selected_metric, 
                                                percentage=skip_percentage, log_x=log_x)
        elif trim_mode == "Fixed Number" and skip_rows is not None:
            fig = create_trimmed_individual_plots(df, selected_runs, selected_metric, 
                                                fixed_rows=skip_rows, log_x=log_x)
        else:
            # Fallback to no trimming
            fig = create_individual_plots(df, selected_runs, selected_metric, log_x)
    
    # Use a key that changes with parameters to force refresh
    plot_key = f"trim_plot_{trim_mode}_{skip_percentage}_{skip_rows}_{view_mode}_{plot_type}"
    st.plotly_chart(fig, use_container_width=True, key=plot_key)
    
    # with col2:
    #     st.subheader("📋 Trimmed Run Info")
    #     
    #     # Show basic stats
    #     st.markdown("**Selected Runs:**")
    #     for run_name in selected_runs:
    #         run_data = df[df['name'] == run_name].iloc[0]
    #         st.markdown(f"• {run_name}")
    #         st.markdown(f"  *State: {run_data['state']}*")
    #     
    #     # Show metric statistics (after trimming)
    #     st.markdown(f"**{selected_metric} Stats (Trimmed):**")
    #     metric_stats = []
    #     
    #     for run_name in selected_runs:
    #         run_data = df[df['name'] == run_name].iloc[0]
    #         hist_df = parse_history(run_data['history'])
    #         
    #         if not hist_df.empty and selected_metric in hist_df.columns:
    #             values = hist_df[selected_metric].dropna()
    #             if not values.empty:
    #                 # Apply trimming
    #                 if trim_mode == "Percentage" and skip_percentage is not None:
    #                     skip_count = int(len(values) * skip_percentage / 100)
    #                 elif trim_mode == "Fixed Number" and skip_rows is not None:
    #                     skip_count = min(skip_rows, len(values) - 1)
    #                 else:
    #                     skip_count = 0
    #                 
    #                 trimmed_values = values.iloc[skip_count:]
    #                 
    #                 if not trimmed_values.empty:
    #                     stats = {
    #                         'Run': run_name,
    #                         'Original': len(values),
    #                         'Trimmed': len(trimmed_values),
    #                         'Skipped': skip_count,
    #                         'Min': f"{trimmed_values.min():.4f}",
    #                         'Max': f"{trimmed_values.max():.4f}",
    #                         'Final': f"{trimmed_values.iloc[-1]:.4f}"
    #                     }
    #                     metric_stats.append(stats)
    #     
    #     if metric_stats:
    #         stats_df = pd.DataFrame(metric_stats)
    #         st.dataframe(stats_df, hide_index=True)
    #     
    #     # Trimming info
    #     st.markdown("**Trimming Info:**")
    #     if trim_mode == "Percentage" and skip_percentage is not None:
    #         st.markdown(f"• Mode: {trim_mode}")
    #         st.markdown(f"• Skipping first {skip_percentage}% of data")
    #     elif trim_mode == "Fixed Number" and skip_rows is not None:
    #         st.markdown(f"• Mode: {trim_mode}")
    #         st.markdown(f"• Skipping first {skip_rows} steps")
    #     else:
    #         st.markdown("• No trimming applied")
    #     st.markdown("• Useful for removing initial outliers")
    #     st.markdown("• Great for grad_norm, learning_rate plots")
    
    # Additional information
    available_metrics = get_available_metrics(df[df['name'].isin(selected_runs)])
    with st.expander("📊 Dataset Information"):
        st.write(f"**Total Runs:** {len(df)}")
        st.write(f"**Selected Runs:** {len(selected_runs)}")
        st.write(f"**Available Metrics:** {len(available_metrics)}")
        st.write("**All Available Metrics:**")
        st.write(", ".join(available_metrics))

if __name__ == "__main__":
    main()