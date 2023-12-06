import pandas as pd 
# Visual comparison (using Matplotlib or Seaborn)
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('examples/explanation_method_comparison_eucliden.csv')

# Statistical summary
print(df.describe())

gausian_blur = df.iloc[lambda x: x.index % 2 == 0]

# Filter for odd rows (indices 1, 3, 5, ...)
gausian_noise = df.iloc[lambda x: x.index % 2 != 0]


sns.boxplot(data=df)
plt.title('Comparison of Explanation Methods')
plt.ylabel('Distance Score')
plt.show()

