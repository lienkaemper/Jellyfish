import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files
mouth_data = pd.read_csv('../data/mouth.csv')
df = pd.read_csv('../data/no_mouth_neurons_full.csv')

print(mouth_data.head())
print(df.head())

mouth_x = mouth_data['x'].values[0]
mouth_y = mouth_data['y'].values[0]

df['x_norm'] = df.x - mouth_x
df['y_norm'] = df.y - mouth_y

fig, ax = plt.subplots(1, figsize = (6,6))
sns.scatterplot(data=df, x='x_norm', y='y_norm', hue='track_id', ax = ax, s = 5, alpha = .5)
plt.axis('equal')


df_0 = df.loc[df.groupby('track_id')['t'].idxmin()][['track_id', 't', 'x_norm', 'y_norm']]

sns.scatterplot(data=df_0, x='x_norm', y='y_norm', hue='track_id', ax = ax)
plt.axis('equal')
plt.show()

df['r'] = np.sqrt(df.x_norm**2 + df.y_norm**2)
df_0['r'] = np.sqrt(df_0.x_norm**2 + df_0.y_norm**2)
df['theta'] = np.arctan2(df.y_norm, df.x_norm)
df_0['theta'] = np.arctan2(df_0.y_norm, df_0.x_norm)
plt.figure(figsize=(6,4))
sns.histplot(df.r, bins = 50, kde = True)
plt.title('Distribution of distances from mouth')
plt.xlabel('Distance from mouth (pixels)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df_0.r, bins = 50, kde = True)
plt.title('Distribution of initial distances from mouth')
plt.xlabel('Initial distance from mouth (pixels)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df.theta, bins = 50, kde = True)
plt.title('Distribution of angles from mouth')
plt.xlabel('Angle from mouth (radians)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df_0.theta, bins = 50, kde = True)
plt.title('Distribution of initial angles from mouth')
plt.xlabel('Initial angle from mouth (radians)')
plt.ylabel('Count')
plt.show()

df['dr'] = df.groupby('track_id')['r'].diff()
df['dtheta'] = df.groupby('track_id')['theta'].diff()
df['dtheta'] = df['dtheta'].apply(lambda x: (x + np.pi) % (2 * np.pi) - np.pi)  # Wrap to [-pi, pi]
plt.figure(figsize=(6,4))
sns.histplot(df.dr, bins = 50, kde = True)
plt.title('Distribution of radial displacements')
plt.xlabel('Radial displacement (pixels)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df.dtheta, bins = 50, kde = True)
plt.title('Distribution of angular displacements')
plt.xlabel('Angular displacement (radians)')
plt.ylabel('Count')
plt.show()

mean_dr = df.dr.mean()
std_dr = df.dr.std()
mean_dtheta = df.dtheta.mean()
std_dtheta = df.dtheta.std()
print(f"Mean radial displacement: {mean_dr}, Std: {std_dr}")
print(f"Mean angular displacement: {mean_dtheta}, Std: {std_dtheta}")

# next goal: create control distribution based on these statistics

n_tracks = df.track_id.nunique()
t_min = df.t.min()
t_max = df.t.max()

birth_times = df_0.t.values
control_tracks = []

for track_id in range(n_tracks):
    birth_time = birth_times[track_id]
    r0 = np.random.choice(df_0.r.values, 1)[0]
    theta0 = 2*np.pi * np.random.rand()
    times = np.arange(birth_time, t_max + 1)
    rs = [r0]
    thetas = [theta0]
    for t in times[1:]:
        dr = np.random.normal(mean_dr, std_dr)
        dtheta = np.random.normal(mean_dtheta, std_dtheta)
        rs.append(rs[-1] + dr)
        thetas.append(thetas[-1] + dtheta)
    xs = [r * np.cos(theta) for r, theta in zip(rs, thetas)]
    ys = [r * np.sin(theta) for r, theta in zip(rs, thetas)]
    track_df = pd.DataFrame({
        'track_id': track_id,
        't': times,
        'x_norm': xs,
        'y_norm': ys, 
        'x' : xs + mouth_x,
        'y' : ys + mouth_y,
        'r': rs,
        'theta': thetas
    })
    control_tracks.append(track_df)

control_df = pd.concat(control_tracks, ignore_index=True)
    
fig, ax = plt.subplots(1, figsize = (6,6))
sns.scatterplot(data=control_df, x='x_norm', y='y_norm', hue='track_id', ax = ax, s = 5, alpha = .5)
plt.axis('equal')
plt.show()