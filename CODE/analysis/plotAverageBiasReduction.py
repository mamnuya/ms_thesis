'''
pip install matplotlib
pip install seaborn
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data from the CSV generated in Step 2
df = pd.read_csv("../../data/lexicon_analysis/bias_reduction/avg_bias_reduction_by_debiasing_method_per_language/cross_language_bias_reduction.csv")

# Set the figure size
plt.figure(figsize=(12, 6))

# Create bar plots for complex and simple debiasing
sns.barplot(x="language", y="avg_complex_reduction_from_original", data=df, color="blue", label="Complex Debiasing")
sns.barplot(x="language", y="avg_simple_reduction_from_original", data=df, color="red", label="Simple Debiasing")

# Customize plot
plt.xlabel("Language")
plt.ylabel("Avg Bias Reduction (%)")
plt.title("Bias Reduction Across Languages")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()

# Show the plot
plt.show()