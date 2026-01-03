import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visual style
sns.set_theme(style="whitegrid")

def generate_metrics_graphs(csv_path):
    """
    Reads a CSV file and generates distribution and box plots for all numeric columns.
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    # Identify numeric columns (metrics)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    base_name = os.path.splitext(csv_path)[0]
    
    print(f"📊 Generating graphs for: {csv_path}")
    
    # 1. Generate Individual Distribution Plots (Histogram + Density)
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, color='teal', bins=15)
        
        plt.title(f'Distribution of {col}\n({base_name})', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        output_name = f"{base_name}_{col.lower()}_dist.png"
        plt.savefig(output_name, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_name}")

    # 2. Generate Combined Boxplots for Summary Statistics
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 6))
    if len(numeric_cols) == 1: axes = [axes] # Handle single column case
        
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i], color='salmon', width=0.4)
        axes[i].set_title(f'{col} Range', fontsize=13)
        axes[i].set_ylabel('') # Clear Y-label for clean look
        
    plt.suptitle(f'Statistical Summary: {base_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    summary_name = f"{base_name}_summary_boxplots.png"
    plt.savefig(summary_name)
    plt.close()
    print(f"   ✅ Saved: {summary_name}")

# --- RUN THE GENERATOR ---
# Replace with your actual file names
files_to_process = [r"C:\Users\arjun\Desktop\hack\n50_snr0.33__final_output_with_enhanced_files\metrics_report_n50_snr0.33.csv"]

for file in files_to_process:
    generate_metrics_graphs(file)