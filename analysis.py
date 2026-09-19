import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = pd.read_csv("train.csv")
train["datetime"] = pd.to_datetime(train["datetime"])
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek # 0=Monday, 6=Sunday

# Check columns to ensure we have 'count', 'weather', 'season', 'dayofweek', 'hour'
print(train.columns)

# Set style
sns.set_theme(style="whitegrid")

# 1. Demand by Weather
plt.figure(figsize=(8, 5))
sns.barplot(data=train, x='weather', y='count', errorbar=None, palette='viridis')
plt.title('Average Rental Demand by Weather Condition')
plt.xlabel('Weather (1: Clear, 2: Mist, 3: Light Rain/Snow, 4: Heavy Rain/Snow)')
plt.ylabel('Average Rentals')


# 2. Demand by Season
plt.figure(figsize=(8, 5))
sns.barplot(data=train, x='season', y='count', errorbar=None, palette='magma')
plt.title('Average Rental Demand by Season')
plt.xlabel('Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)')
plt.ylabel('Average Rentals')


# 3. Demand by Day of Week
plt.figure(figsize=(8, 5))
sns.barplot(data=train, x='dayofweek', y='count', errorbar=None, palette='coolwarm')
plt.title('Average Rental Demand by Day of the Week')
plt.xlabel('Day of Week (0: Monday, 6: Sunday)')
plt.ylabel('Average Rentals')


# 4. Demand by Hour
plt.figure(figsize=(10, 5))
sns.lineplot(data=train, x='hour', y='count', estimator='mean', errorbar=None, marker='o', color='b')
plt.title('Average Rental Demand by Hour of the Day')
plt.xlabel('Hour of the Day (0-23)')
plt.ylabel('Average Rentals')
plt.xticks(range(0, 24))
plt.grid(True)


# 5. Demand by Hour grouped by Working Day
plt.figure(figsize=(10, 5))
sns.lineplot(data=train, x='hour', y='count', hue='workingday', estimator='mean', errorbar=None, marker='o')
plt.title('Hourly Rental Demand: Working Day vs Non-Working Day')
plt.xlabel('Hour of the Day (0-23)')
plt.ylabel('Average Rentals')
plt.xticks(range(0, 24))
plt.grid(True)


plt.show()

# Calculate some stats to return
weather_stats = train.groupby('weather')['count'].mean().to_dict()
season_stats = train.groupby('season')['count'].mean().to_dict()
print(f"Weather means: {weather_stats}")
print(f"Season means: {season_stats}")