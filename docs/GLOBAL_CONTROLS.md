# Global Controls Implementation

## Changes Made

### 🌐 Centralized Sidebar Controls
**Before:**
- Separate controls for Standard View and Trimmed View
- Duplicate run/metric selection in each tab
- Different keys for each control set

**After:**
- Single set of global controls in sidebar
- Controls apply to both Standard and Trimmed views
- No duplication, cleaner interface

### 📋 Global Control Sections

#### 1. **Select Runs** (Global)
- Choose which runs to visualize
- "Select All" option
- Applies to both views

#### 2. **Select Metric** (Global)
- Choose metric to visualize
- Auto-detects available metrics
- Works for both Standard and Trimmed views

#### 3. **Plot Options** (Global)
- Plot Type: Line or Scatter
- View Mode: Comparison or Individual
- Consistent across both tabs

#### 4. **Trimming Options** (Global)
- Trimming Mode: Percentage or Fixed Number
- Percentage: 0-100% slider
- Fixed Number: 0-1000 steps input
- Individual step counts shown for each run
- Only affects Trimmed View

### 🔧 Function Changes

#### Updated Function Signatures:
```python
# Before
def show_standard_view(df):
def show_trimmed_view(df):

# After  
def show_standard_view(df, selected_runs, selected_metric, plot_type, view_mode):
def show_trimmed_view(df, selected_runs, selected_metric, plot_type, view_mode, 
                     trim_mode, skip_percentage, skip_rows):
```

#### Removed Duplicate Code:
- ❌ Removed duplicate run selection from each view
- ❌ Removed duplicate metric selection from each view  
- ❌ Removed duplicate plot options from each view
- ❌ Removed duplicate trimming controls from trimmed view function

### 🎯 Benefits

1. **Consistent Experience:**
   - Same selections apply to both views
   - Switch between tabs without losing settings
   - No need to reconfigure for each view

2. **Cleaner Interface:**
   - Single sidebar with all controls
   - No duplicate controls
   - Less visual clutter

3. **Better Workflow:**
   - Select runs/metrics once, view in multiple ways
   - Easy comparison between standard and trimmed views
   - Trimming options always visible (even if not used)

4. **Individual Step Counts:**
   - Shows exact steps that will be skipped per run
   - More precise than average calculations
   - Better insight into data variations

## App Location
Updated app running at: **http://localhost:8507**

## Current Interface Structure
```
Sidebar:
├── 🎛️ Global Controls
│   ├── Select Runs
│   ├── Select Metric
│   ├── Plot Options (Type, View Mode)
│   └── ✂️ Trimming Options
│
Main Area:
├── 📈 Standard View Tab
└── ✂️ Trimmed View Tab
```

The interface is now much more streamlined with global controls that work across both view types!