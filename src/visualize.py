import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations(results_dir: str = "results/raw", output_dir: str = "results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(f"{results_dir}/*.json")
    if not files:
        print("No result files found.")
        return
        
    models = []
    sfr_scores = []
    
    for f in files:
        with open(f, "r") as file:
            data = json.load(file)
            
            # Exclude models with 0 successful runs (where errors == n_examples)
            if data.get("n_examples", 0) - data.get("errors", 0) > 0:
                models.append(data["model_id"].split("/")[-1])
                sfr_scores.append(data["sfr"])
                
    if not models:
        print("No valid data to plot (all runs had 0 successes).")
        return
            
    # Sort by SFR descending
    sorted_pairs = sorted(zip(sfr_scores, models), reverse=True)
    sfr_scores, models = zip(*sorted_pairs)
    
    # 1. Bar Chart: Separation Failure Rate per Model
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(x=list(models), y=list(sfr_scores), palette="rocket")
    plt.title("Separation Failure Rate (SFR) in Agentic Workflows", fontsize=16, pad=20)
    plt.ylabel("Failure Rate (%)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    
    # Add percentage labels on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 9), 
                   textcoords='offset points',
                   fontsize=11, fontweight='bold')
                   
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sfr_results.png", dpi=300)
    print(f"Saved {output_dir}/sfr_results.png")

if __name__ == "__main__":
    generate_visualizations()
