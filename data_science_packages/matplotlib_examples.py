import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('MacOSX')

# Creating simple line plot
x = np.linspace(0, 10, 100)  # 100 points from 0 to 10
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Simple Line Plot")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show(block=True)

# Creating scatter plot
x = np.random.rand(100)
y = np.random.rand(100)

plt.figure()
plt.scatter(x, y)
plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show(block=True)

# Creating bar plot
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]

plt.figure()
plt.bar(categories, values, color='skyblue')
plt.title("Bar Plot")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.grid(True, axis='y')
plt.show(block=True)

# Creating histogram
data = np.random.randn(1000)  # 1000 random values

plt.figure()
plt.hist(data, bins=30, color='purple', edgecolor='black')
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show(block=True)

# Creating pie chart
sizes = [15, 30, 45, 10]
labels = ['A', 'B', 'C', 'D']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice

plt.figure()
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Pie Chart")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show(block=True)

# Creating subplot
x = np.linspace(0, 2 * np.pi, 400)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y1)
plt.title("Sine")

plt.subplot(1, 2, 2)
plt.plot(x, y2)
plt.title("Cosine")

plt.suptitle("Subplots Example")
plt.show(block=True)

# Creating a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

ax.plot_surface(x, y, z, cmap='viridis')

plt.title("3D Surface Plot")
plt.show(block=True)

# Customizing plots
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure()
plt.plot(x, y1, label='sin(x)', color='blue', linestyle='--')
plt.plot(x, y2, label='cos(x)', color='red', linestyle='-.')

plt.title("Customized Plot")
plt.xlabel("x")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show(block=True)

# Saving a figure
plt.figure()
plt.plot(x, y1, label='sin(x)')
plt.title("Plot to Save")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.grid(True)
plt.savefig('saved_examples/saved_plot.png')  # Save the plot as a PNG file
print("Plot saved as 'saved_plot.png'")
