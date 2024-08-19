import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Loading an example dataset
tips = px.data.tips()

# Scatter plot with trendline
fig = px.scatter(tips, x='total_bill', y='tip', trendline='ols', title='Scatter Plot with Trendline')
fig.show()

# Bar plot
fig = px.bar(tips, x='day', y='total_bill', title='Bar Plot of Total Bill by Day')
fig.show()

# Box plot
fig = px.box(tips, x='day', y='total_bill', points="all", title='Box Plot of Total Bill by Day')
fig.show()

# Violin plot
fig = px.violin(tips, x='day', y='total_bill', title='Violin Plot of Total Bill by Day')
fig.show()

# Histogram
fig = px.histogram(tips, x='total_bill', nbins=30, title='Histogram of Total Bill')
fig.show()

# Heatmap (correlation matrix)
corr = tips[['total_bill', 'tip', 'size']].corr()
fig = go.Figure(data=go.Heatmap(z=corr.values,
                                x=corr.index.values,
                                y=corr.columns.values,
                                colorscale='Viridis',
                                colorbar=dict(title='Correlation')))
fig.update_layout(title='Heatmap of Correlation Matrix', height=400, width=400)
fig.show()

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
fig.update_layout(title='Line Plot of sin(x)', xaxis_title='x', yaxis_title='sin(x)')
fig.show()

# 3D Scatter plot
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species', title='3D Scatter Plot')
fig.show()

# Parallel Coordinates plot
df = px.data.iris()
fig = px.parallel_coordinates(df, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], color="species_id",
                              title='Parallel Coordinates Plot')
fig.show()

# Bubble chart
df = px.data.gapminder()
fig = px.scatter(df.query("year==2007"), x='gdpPercap', y='lifeExp', size='pop', color='continent', hover_name='country',
                 title='Bubble Chart of GDP vs Life Expectancy (2007)')
fig.show()

# Sunburst chart
fig = px.sunburst(df, path=['continent', 'country'], values='pop', color='lifeExp', hover_data=['iso_alpha'],
                  title='Sunburst Chart of Population by Continent and Country')
fig.show()

