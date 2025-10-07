# Interface Updates Summary

## Changes Made

### 1. Extended Percentage Range
- **Before**: Slider limited to 0-50%
- **After**: Slider now allows 0-100%
- **Benefit**: Can trim even more data for extreme outlier cases

### 2. Updated Terminology: "Rows" → "Steps" 
Updated throughout the interface to use more accurate terminology:

#### Sidebar Controls:
- "Skip first X rows:" → "Skip first X steps:"
- Info text: "~{X} rows on average" → "~{X} steps on average"

#### Plot Titles:
- "First {X} Rows Trimmed" → "First {X} Steps Trimmed"
- Subtitles updated accordingly

#### Statistics & Info Panels:
- "Skipping first {X} rows" → "Skipping first {X} steps"
- All references to "rows" changed to "steps"

## Why These Changes?

### Extended Range (0-100%)
- **Use Case**: Some metrics like gradient norm can have extreme outliers in the first 60-80% of training
- **Flexibility**: Allows complete control over data visualization
- **Edge Cases**: Can now handle datasets where most of the interesting data is at the end

### "Steps" vs "Rows" Terminology
- **Accuracy**: Training data is measured in steps/iterations, not database rows
- **Clarity**: More intuitive for ML practitioners
- **Consistency**: Aligns with standard ML terminology (training steps, global steps, etc.)

## App Location
The updated app is running at: **http://localhost:8504**

## Features Now Available
✅ Percentage trimming: 0-100% (was 0-50%)
✅ Fixed step trimming: 0-1000 steps
✅ Consistent "steps" terminology throughout
✅ Both x-axis and y-axis properly trimmed
✅ Real-time updates when changing parameters