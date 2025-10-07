# WandB Runs Visualizer

A Streamlit app for quickly visualizing and comparing metrics across multiple WandB runs.

## Features

### 📈 Standard View
- **Multi-run Comparison**: Select multiple runs and compare their metrics on the same plot
- **Individual Plots**: View metrics for each run in separate subplots
- **Interactive Controls**: 
  - Select runs to compare
  - Choose from all available metrics
  - Switch between line and scatter plots
  - Toggle between comparison and individual view modes
- **Statistics**: View min, max, final values, and data points for each metric
- **Auto-detection**: Automatically detects all available metrics from your runs

### ✂️ Trimmed View (NEW!)
- **Data Trimming**: Remove initial data points that might be outliers or not interesting
- **Flexible Trimming Options**:
  - **Percentage Mode**: Skip first X% of data points (default: 10%)
  - **Fixed Rows Mode**: Skip first X rows (configurable)
- **Perfect for**: Grad norm plots, learning rate schedules, or any metric with large initial values
- **Enhanced Statistics**: Shows original vs trimmed data point counts
- **Same Visualization Options**: All plotting features from standard view

## Usage

1. Make sure your `wandb_runs_data.csv` file is in the same directory
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run streamlit_wandb_visualizer.py`
4. Open your browser to the displayed URL (usually `http://localhost:8501` or `8502`)

## Controls

### 📈 Standard View Tab
#### Sidebar
- **Select Runs**: Choose which runs to visualize (use "Select All" for convenience)
- **Select Metric**: Choose from automatically detected metrics (e.g., train/loss, eval/loss)
- **Plot Type**: Line or scatter plot
- **View Mode**: 
  - **Comparison**: All runs on the same plot for easy comparison
  - **Individual**: Separate subplot for each run

### ✂️ Trimmed View Tab  
#### Sidebar
- **Select Runs**: Same as standard view
- **Select Metric**: Same as standard view (defaults to train/grad_norm if available)
- **Trimming Options**:
  - **Percentage Mode**: Skip first X% of data points (slider: 0-50%)
  - **Fixed Number Mode**: Skip first X rows (input: 0-1000)
- **Plot Type & View Mode**: Same as standard view

#### When to Use Trimmed View
- **Gradient Norm**: Often has very high initial values that make the rest hard to see
- **Learning Rate**: Warmup periods might not be interesting
- **Loss Spikes**: Remove initial training instability
- **Any metric**: Where early values are outliers or not representative

### Main View
- Interactive Plotly charts with zoom, pan, and hover
- Enhanced statistics panel showing trimming effects
- Run information and dataset details

## Common Metrics
The app automatically detects metrics like:
- `train/loss`
- `eval/loss` 
- `train/learning_rate`
- `train/global_step`
- And many more from your WandB runs

## Tips
### General
- Use "Comparison" mode to easily compare runs side-by-side
- Use "Individual" mode to see detailed trends for each run
- Check the statistics panel for quick numerical comparisons
- The app caches data for better performance

### Trimmed View Specific
- **Start with 10% trimming** for most metrics to remove initial noise
- **Use 20-30% for grad_norm** which often has very high initial values
- **Try Fixed Number mode** when you know exactly how many initial steps to skip
- **Compare with Standard View** to see the difference trimming makes
- **Perfect for publications** - cleaner plots without initial outliers