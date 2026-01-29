import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.linear_model import LinearRegression

def sample_from_linear_distribution(n_samples, slope, intercept, min_dist, max_dist):
    """
    Sample from the distribution p(r) = (slope * r + intercept) / norm_factor
    using inverse transform sampling.
    """
    # Generate uniform random samples
    u = np.random.uniform(0, 1, n_samples)
    
    # Inverse CDF: solve for r given u
    # CDF(r) = integral of p(r) from min_dist to r
    # For linear p(r) = (slope*r + intercept)/norm_factor:
    # CDF(r) = (slope/2 * (r^2 - min_dist^2) + intercept * (r - min_dist)) / norm_factor
    
    # Setting CDF(r) = u and solving the quadratic equation:
    # (slope/2) * r^2 + intercept * r - (slope/2 * min_dist^2 + intercept * min_dist + u * norm_factor) = 0
    
    #normalizing factor: 
    norm_factor = (slope/2) * (max_dist**2 - min_dist**2) + intercept * (max_dist - min_dist)


    a = slope / 2
    b = intercept
    c = -(slope / 2 * min_dist**2 + intercept * min_dist + u * norm_factor)
    
    # Quadratic formula: r = (-b ± sqrt(b^2 - 4ac)) / 2a
    discriminant = b**2 - 4*a*c
    r_samples = (-b + np.sqrt(discriminant)) / (2*a)
    
    return r_samples



parser = argparse.ArgumentParser()
parser.add_argument('--show_plot', action='store_true', help='Generate plots')
args = parser.parse_args()
plot = args.show_plot

# Load the CSV files
mouth_data = pd.read_csv('../data/mouth.csv')
df = pd.read_csv('../data/no_mouth_neurons_full.csv')



mouth_x = mouth_data['x'].values[0]
mouth_y = mouth_data['y'].values[0]

df['x_norm'] = df.x - mouth_x
df['y_norm'] = df.y - mouth_y



df_0 = df.loc[df.groupby('track_id')['t'].idxmin()][['track_id', 't', 'x_norm', 'y_norm']]




df['r'] = np.sqrt(df.x_norm**2 + df.y_norm**2)
df_0['r'] = np.sqrt(df_0.x_norm**2 + df_0.y_norm**2)
df['theta'] = np.arctan2(df.y_norm, df.x_norm)
df_0['theta'] = np.arctan2(df_0.y_norm, df_0.x_norm)

counts, bin_edges = np.histogram(df_0.r, bins=50)
midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])
model = LinearRegression()
X = midpoints.reshape(-1, 1)
y = counts
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_

preds = model.predict(X)

plt.figure(figsize=(6,4))
sns.histplot(df.r, bins = 50, kde = True)
plt.title('Distribution of distances from mouth')
plt.xlabel('Distance from mouth (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/distance_distribution.pdf')
if plot:
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df_0.r, bins = 50, kde = True)
plt.plot(midpoints, preds, color='red', linestyle='--', label='Linear Fit')
plt.title('Distribution of initial distances from mouth')
plt.xlabel('Initial distance from mouth (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/initial_distance_distribution.pdf')
if plot:
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df.theta, bins = 50, kde = True)
plt.title('Distribution of angles from mouth')
plt.xlabel('Angle from mouth (radians)')
plt.ylabel('Count')
plt.savefig('../results/plots/angle_distribution.pdf')
if plot:
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df_0.theta, bins = 50, kde = True)
plt.title('Distribution of initial angles from mouth')
plt.xlabel('Initial angle from mouth (radians)')
plt.ylabel('Count')
plt.savefig('../results/plots/initial_angle_distribution.pdf')
if plot:
    plt.show()

df['dr'] = df.groupby('track_id')['r'].diff()
df['dtheta'] = df.groupby('track_id')['theta'].diff()
df['dtheta'] = df['dtheta'].apply(lambda x: (x + np.pi) % (2 * np.pi) - np.pi)  # Wrap to [-pi, pi]
mean_dr = df.dr.mean()
std_dr = df.dr.std()
mean_dtheta = df.dtheta.mean()
std_dtheta = df.dtheta.std()

plt.figure(figsize=(4,4))
sns.histplot(df.dr, bins = 50, kde = True)
plt.axvline(x=mean_dr, color='red', linestyle='--', label='Mean')
plt.title('Distribution of radial displacements')
plt.xlabel('Radial displacement (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/radial_displacement_distribution.pdf')
if plot:
    plt.show()

plt.figure(figsize=(4,4))
sns.histplot(df.dtheta, bins = 50, kde = True)
plt.axvline(x=mean_dtheta, color='red', linestyle='--', label='Mean')
plt.title('Distribution of angular displacements')
plt.xlabel('Angular displacement (radians)')
plt.ylabel('Count')
plt.savefig('../results/plots/angular_displacement_distribution.pdf')
if plot:
    plt.show()




# next goal: create control distribution based on these statistics

n_tracks = df.track_id.nunique()
t_min = df.t.min()
t_max = df.t.max()

birth_times = df_0.t.values


def create_control_df(birth_radii, birth_angles, birth_times, mean_dr, std_dr, mean_dtheta, std_dtheta, t_max):

    control_tracks = []

    for track_id in range(n_tracks):
        birth_time = birth_times[track_id]
        r0 = birth_radii[track_id]
        theta0 = birth_angles[track_id]
        
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
    return control_df





fig, ax = plt.subplots(1, figsize = (6,6))
sns.scatterplot(data=df, x='x_norm', y='y_norm', hue='track_id', ax = ax, s = 5, alpha = .5)
sns.scatterplot(data=df_0, x='x_norm', y='y_norm', hue='track_id', ax = ax)
plt.title("Original Tracks")
plt.axis('equal')
plt.savefig('../results/plots/raw_tracks_data.pdf')
plt.savefig('../results/plots/raw_tracks_data.png')
if plot:
    plt.show()

r0 = np.random.choice(df_0.r.values, size = n_tracks)
theta0 = 2*np.pi * np.random.rand(n_tracks)
control_df_data_r = create_control_df(r0, theta0, birth_times, mean_dr, std_dr, mean_dtheta, std_dtheta, t_max)
control_df_data_r_0 = control_df_data_r.loc[control_df_data_r.groupby('track_id')['t'].idxmin()][['track_id', 't', 'x_norm', 'y_norm', 'r']]

fig, ax = plt.subplots(1, figsize = (6,6))
sns.scatterplot(data=control_df_data_r, x='x_norm', y='y_norm', hue='track_id', ax = ax, s = 5, alpha = .5)
sns.scatterplot(data=control_df_data_r_0, x='x_norm', y='y_norm', hue='track_id', ax = ax)
plt.title("Control Tracks (Data r0)")
plt.axis('equal')
plt.savefig('../results/plots/control_tracks_data.pdf')
plt.savefig('../results/plots/control_tracks_data.png')
if plot:
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(control_df_data_r_0.r, bins = 50, kde = True, color='blue', alpha = .5, label='Control (shuffled)')
sns.histplot(df_0.r, bins = 50, kde = True, color='orange', alpha = .5, label='Original')
plt.title('Distribution of initial distances from mouth: shuffle control vs. true')
plt.xlabel('Initial distance from mouth (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/initial_distance_distribution_shuffle.pdf')
if plot:
    plt.show()

r0 = np.random.choice(df_0.r.values, size = n_tracks)
theta0 = 2*np.pi * np.random.rand(n_tracks)
control_df_data_r_motion_off = create_control_df(r0, theta0, birth_times, 0, 0, 0, 0, t_max)
control_df_data_r_motion_off_0 = control_df_data_r_motion_off.loc[control_df_data_r_motion_off.groupby('track_id')['t'].idxmin()][['track_id', 't', 'x_norm', 'y_norm', 'r']]

fig, ax = plt.subplots(1, figsize = (6,6))
sns.scatterplot(data=control_df_data_r_motion_off, x='x_norm', y='y_norm', hue='track_id', ax = ax, s = 5, alpha = .5)
sns.scatterplot(data=control_df_data_r_motion_off_0, x='x_norm', y='y_norm', hue='track_id', ax = ax)
plt.title("Control Tracks (Data r0)")
plt.axis('equal')
plt.savefig('../results/plots/control_tracks_data_motion_off.pdf')
plt.savefig('../results/plots/control_tracks_data_motion_off.png')
if plot:
    plt.show()



r0 = sample_from_linear_distribution(n_tracks, slope, intercept, df_0.r.min(), df_0.r.max())
theta0 = 2*np.pi * np.random.rand(n_tracks)
control_df_lin_samples = create_control_df(r0, theta0, birth_times, mean_dr, std_dr, mean_dtheta, std_dtheta, t_max)
control_df_lin_samples_0 = control_df_lin_samples.loc[control_df_lin_samples.groupby('track_id')['t'].idxmin()][['track_id', 't', 'x_norm', 'y_norm', 'r']]

fig, ax = plt.subplots(1, figsize = (6,6))
sns.scatterplot(data=control_df_lin_samples, x='x_norm', y='y_norm', hue='track_id', ax = ax, s = 5, alpha = .5)    
sns.scatterplot(data=control_df_lin_samples_0, x='x_norm', y='y_norm', hue='track_id', ax = ax)
plt.title("Control Tracks (Linear distributed r0)")
plt.axis('equal')
plt.savefig('../results/plots/control_tracks_linear_samples_data.pdf')
plt.savefig('../results/plots/control_tracks_linear_samples_data.png')
if plot:
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(control_df_lin_samples_0.r, bins = 50, kde = True, color='blue', alpha = .5, label='Control (linear sample)')
sns.histplot(df_0.r, bins = 50, kde = True, color='orange', alpha = .5, label='Original')
plt.title('Distribution of initial distances from mouth: linear distributed control vs. true')
plt.xlabel('Initial distance from mouth (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/initial_distance_distribution_linear.pdf')
if plot:
    plt.show()


r0 = sample_from_linear_distribution(n_tracks, slope, intercept, df_0.r.min(), df_0.r.max())
theta0 = 2*np.pi * np.random.rand(n_tracks)
control_df_lin_samples_motion_off = create_control_df(r0, theta0, birth_times, 0, 0, 0, 0, t_max)
control_df_lin_samples_motion_off_0 = control_df_lin_samples_motion_off.loc[control_df_lin_samples_motion_off.groupby('track_id')['t'].idxmin()][['track_id', 't', 'x_norm', 'y_norm', 'r']]

fig, ax = plt.subplots(1, figsize = (6,6))
sns.scatterplot(data=control_df_lin_samples_motion_off, x='x_norm', y='y_norm', hue='track_id', ax = ax, s = 5, alpha = .5)    
sns.scatterplot(data=control_df_lin_samples_motion_off_0, x='x_norm', y='y_norm', hue='track_id', ax = ax)
plt.title("Control Tracks (Linear distributed r0)")
plt.axis('equal')
plt.savefig('../results/plots/control_tracks_linear_samples_data_motion_off.pdf')
plt.savefig('../results/plots/control_tracks_linear_samples_data_motion_off.png')
if plot:
    plt.show()


r0 = sample_from_linear_distribution(n_tracks, 1, 0, df_0.r.min(), df_0.r.max())
theta0 = 2*np.pi * np.random.rand(n_tracks)
control_df_uniform = create_control_df(r0, theta0, birth_times, mean_dr, std_dr, mean_dtheta, std_dtheta, t_max)
control_df_uniform_0 = control_df_uniform.loc[control_df_uniform.groupby('track_id')['t'].idxmin()][['track_id', 't', 'x_norm', 'y_norm', 'r']]

fig, ax = plt.subplots(1, figsize = (6,6))
sns.scatterplot(data=control_df_uniform, x='x_norm', y='y_norm', hue='track_id', ax = ax, s = 5, alpha = .5)    
sns.scatterplot(data=control_df_uniform_0, x='x_norm', y='y_norm', hue='track_id', ax = ax)
plt.title("Control Tracks (Uniform distributed r0)")
plt.axis('equal')
plt.savefig('../results/plots/control_tracks_uniform_samples_data.pdf')
plt.savefig('../results/plots/control_tracks_uniform_samples_data.png')
if plot:
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(control_df_uniform_0.r, bins = 50, kde = True, color='blue', alpha = .5, label='Control uniform')
sns.histplot(df_0.r, bins = 50, kde = True, color='orange', alpha = .5, label='Original')
plt.title('Distribution of initial distances from mouth: uniform control vs. true')
plt.xlabel('Initial distance from mouth (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/initial_distance_distribution_uniform.pdf')
if plot:
    plt.show()

r0 = sample_from_linear_distribution(n_tracks, 1, 0, df_0.r.min(), df_0.r.max())
theta0 = 2*np.pi * np.random.rand(n_tracks)
control_df_uniform_motion_off = create_control_df(r0, theta0, birth_times, 0, 0, 0, 0, t_max)
control_df_uniform_motion_off_0 = control_df_uniform_motion_off.loc[control_df_uniform_motion_off.groupby('track_id')['t'].idxmin()][['track_id', 't', 'x_norm', 'y_norm', 'r']]
fig, ax = plt.subplots(1, figsize = (6,6))

sns.scatterplot(data=control_df_uniform_motion_off, x='x_norm', y='y_norm', hue='track_id', ax = ax, s = 5, alpha = .5)    
sns.scatterplot(data=control_df_uniform_motion_off_0, x='x_norm', y='y_norm', hue='track_id', ax = ax)
plt.title("Control Tracks (Uniform distributed r0)")
plt.axis('equal')
plt.savefig('../results/plots/control_tracks_uniform_samples_data_motion_off.pdf')
plt.savefig('../results/plots/control_tracks_uniform_samples_data_motion_off.png')
if plot:
    plt.show()  

# for each track, find the distance to the nearest neigbor at birth time 
def find_nearest_neighbor_distances(df):
    birth_positions_and_times = df.loc[df.groupby('track_id')['t'].idxmin()][['track_id','x_norm', 'y_norm', 't']].values
    nearest_distances = []
    for track_id, x, y, t in birth_positions_and_times:
        positions_time_t = df.loc[df.t == t][['track_id', 'x_norm', 'y_norm']].values
        distances = np.sqrt((positions_time_t[:,1] - x)**2 + (positions_time_t[:,2] - y)**2)
        distances[positions_time_t[:,0] == track_id]  = np.inf  # Exclude self-distance  
        nearest_neigbor = np.argmin(distances)
        nearest_distance = distances[nearest_neigbor]
        nearest_distances.append(nearest_distance)
    return nearest_distances

nearest_distances_control_data_r = find_nearest_neighbor_distances(control_df_data_r)
nearest_distances_control_data_r_motion_off = find_nearest_neighbor_distances(control_df_data_r_motion_off)
nearest_distances_control_lin_samples = find_nearest_neighbor_distances(control_df_lin_samples)
nearest_distances_control_lin_samples_motion_off = find_nearest_neighbor_distances(control_df_lin_samples_motion_off)
nearest_distances_control_uniform = find_nearest_neighbor_distances(control_df_uniform)
nearest_distances_control_uniform_motion_off = find_nearest_neighbor_distances(control_df_uniform_motion_off)
nearest_distances = find_nearest_neighbor_distances(df)

plt.figure(figsize=(6,4))
sns.histplot(nearest_distances_control_data_r, bins = np.linspace(0, 3500, 100), kde = True, label='Control (data r0, shuffled)', stat = 'density', color='blue', alpha = .5)
sns.histplot(nearest_distances, bins = np.linspace(0, 3500, 100), kde = True, color='orange', label='Original', stat = 'density', alpha = .5)
plt.legend()
plt.title('Distribution of nearest neighbor distances at birth')
plt.xlabel('Nearest neighbor distance (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/nearest_neighbor_distance_distribution_shuffle.pdf')
if plot:
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(nearest_distances_control_data_r_motion_off, bins = np.linspace(0, 3500, 100), kde = True, label='Control (data r0, shuffled)', stat = 'density', color='blue', alpha = .5)
sns.histplot(nearest_distances, bins = np.linspace(0, 3500, 100), kde = True, color='orange', label='Original', stat = 'density', alpha = .5)
plt.legend()
plt.title('Distribution of nearest neighbor distances at birth')
plt.xlabel('Nearest neighbor distance (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/nearest_neighbor_distance_distribution_shuffle_motion_off.pdf')
if plot:
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(nearest_distances_control_lin_samples, bins = np.linspace(0, 3500, 100), kde = True, label='Control (linear sample)', stat = 'density', color='blue', alpha = .5)
sns.histplot(nearest_distances, bins = np.linspace(0, 3500, 100), kde = True, color='orange', label='Original', stat = 'density', alpha = .5)
plt.legend()
plt.title('Distribution of nearest neighbor distances at birth')
plt.xlabel('Nearest neighbor distance (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/nearest_neighbor_distance_distribution_linear.pdf')
if plot:
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(nearest_distances_control_lin_samples_motion_off, bins = np.linspace(0, 3500, 100), kde = True, label='Control (linear sample)', stat = 'density', color='blue', alpha = .5)
sns.histplot(nearest_distances, bins = np.linspace(0, 3500, 100), kde = True, color='orange', label='Original', stat = 'density', alpha = .5)
plt.legend()
plt.title('Distribution of nearest neighbor distances at birth')
plt.xlabel('Nearest neighbor distance (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/nearest_neighbor_distance_distribution_linear_motion_off.pdf')
if plot:
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(nearest_distances_control_uniform, bins = np.linspace(0, 3500, 100), kde = True, label='Control (uniform sample)', stat = 'density', color='blue', alpha = .5)
sns.histplot(nearest_distances, bins = np.linspace(0, 3500, 100), kde = True, color='orange', label='Original', stat = 'density', alpha = .5)
plt.legend()
plt.title('Distribution of nearest neighbor distances at birth')
plt.xlabel('Nearest neighbor distance (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/nearest_neighbor_distance_distribution_uniform.pdf')
if plot:
    plt.show()

plt.figure(figsize=(6,4))
sns.histplot(nearest_distances_control_uniform_motion_off, bins = np.linspace(0, 3500, 100), kde = True, label='Control (uniform sample)', stat = 'density', color='blue', alpha = .5)
sns.histplot(nearest_distances, bins = np.linspace(0, 3500, 100), kde = True, color='orange', label='Original', stat = 'density', alpha = .5)
plt.legend()
plt.title('Distribution of nearest neighbor distances at birth')
plt.xlabel('Nearest neighbor distance (pixels)')
plt.ylabel('Count')
plt.savefig('../results/plots/nearest_neighbor_distance_distribution_uniform_motion_off.pdf')
if plot:
    plt.show()