# Importing seaborn and other necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Setting the aesthetic style of the plots
sns.set(style="darkgrid")

# Loading an example dataset
tips = sns.load_dataset("tips")
print("Tips dataset head:\n", tips.head())

# Creating a simple scatter plot
plt.figure()
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.title("Scatter Plot of Total Bill vs Tip")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.show(block=True)

# Creating a histogram
plt.figure()
sns.histplot(tips["total_bill"], bins=30, kde=True)
plt.title("Histogram of Total Bill")
plt.xlabel("Total Bill")
plt.ylabel("Frequency")
plt.show(block=True)

# Creating a bar plot
plt.figure()
sns.barplot(x="day", y="total_bill", data=tips, ci=None)
plt.title("Bar Plot of Total Bill by Day")
plt.xlabel("Day")
plt.ylabel("Total Bill")
plt.show(block=True)

# Creating a box plot
plt.figure()
sns.boxplot(x="day", y="total_bill", data=tips)
plt.title("Box Plot of Total Bill by Day")
plt.xlabel("Day")
plt.ylabel("Total Bill")
plt.show(block=True)

# Creating a violin plot
plt.figure()
sns.violinplot(x="day", y="total_bill", data=tips)
plt.title("Violin Plot of Total Bill by Day")
plt.xlabel("Day")
plt.ylabel("Total Bill")
plt.show(block=True)

# Creating a pair plot
plt.figure()
sns.pairplot(tips)
plt.suptitle("Pair Plot of Tips Dataset", y=1.02)
plt.show(block=True)

# Perform label encoding for categorical variables
label_encoder = LabelEncoder()
tips["sex_encoded"] = label_encoder.fit_transform(tips["sex"])
tips["smoker_encoded"] = label_encoder.fit_transform(tips["smoker"])
tips["day_encoded"] = label_encoder.fit_transform(tips["day"])
tips["time_encoded"] = label_encoder.fit_transform(tips["time"])

# Creating a heatmap
corr = tips[['total_bill', 'tip', 'sex_encoded', 'smoker_encoded', 'day_encoded', 'time_encoded', 'size']].corr()
plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Heatmap of Correlation Matrix")
plt.show(block=True)

# Creating a joint plot
plt.figure()
sns.jointplot(x="total_bill", y="tip", data=tips, kind="hex", color="k")
plt.suptitle("Joint Plot of Total Bill and Tip", y=1.02)
plt.show(block=True)

# Creating a lmplot (linear model plot)
plt.figure()
sns.lmplot(x="total_bill", y="tip", data=tips)
plt.title("Linear Model Plot of Total Bill vs Tip")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.show(block=True)

# Creating a facet grid
g = sns.FacetGrid(tips, col="time")
g.map(sns.histplot, "total_bill")
g.set_titles("Total Bill by {col_name}")
plt.show(block=True)

# Customizing plots
plt.figure()
sns.scatterplot(x="total_bill", y="tip", data=tips, hue="day", style="time", size="size", palette="deep")
plt.title("Customized Scatter Plot of Total Bill vs Tip")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.legend(title="Day/Time/Size")
plt.show(block=True)

# Saving a Seaborn plot
plt.figure()
sns.boxplot(x="day", y="total_bill", data=tips)
plt.title("Box Plot of Total Bill by Day")
plt.xlabel("Day")
plt.ylabel("Total Bill")
plt.savefig('saved_examples/seaborn_boxplot.png')  # Save the plot as a PNG file
print("Plot saved as 'seaborn_boxplot.png'")
