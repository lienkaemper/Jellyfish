using DelimitedFiles, DataFrames
using Plots, StatsPlots

data, header = readdlm("data/Quadrant_lower_right.csv", ',', header=true)
df = DataFrame(data,vec(header))

mouth_x = df[df.track_id .== 98, :x][1]
mouth_y = df[df.track_id .== 98, :y][1]

df.x_corr = df.x .- mouth_x
df.y_corr = df.y .- mouth_y
df.r = sqrt.(df.x_corr .^ 2 .+ df.y_corr .^ 2)
df.theta = (180/π) * atan.(df.y_corr, df.x_corr)

@df df histogram(:r, legend=false,
    title="Distance from mouth", xlabel="Distance (pixels)", ylabel="Count")

@df df histogram(:theta, 
    bins = 0:1.5:90,
    legend=false,
    title="Angle from mouth", 
    xlabel="Angle (radians)", 
    ylabel="Count")   
    
savefig("results/plots/distance_angle_histograms.pdf")   

start_end_locs = combine(groupby(df, :track_id), 
    :x_corr => first => :start_x_corr,
    :y_corr => first => :start_y_corr,
    :r => first => :start_r,
    :theta => first => :start_theta,
    :x_corr => last => :end_x_corr,
    :y_corr => last => :end_y_corr,
    :r => last => :end_r,
    :theta => last => :end_theta
)

@df start_end_locs histogram(:start_r, 
    bins = 0:100:maximum(start_end_locs.start_r),
    legend=false,
    alpha = 0.5,
    label="Starting Distance from mouth", 
    xlabel="Distance (pixels)", 
    ylabel="Count")

@df start_end_locs histogram!(:end_r, 
    bins = 0:100:maximum(start_end_locs.end_r),
    legend=true,
    alpha = 0.5,
    label="Ending Distance from mouth", 
    xlabel="Distance (pixels)", 
    ylabel="Count") 
    
savefig("results/plots/start_end_distance_histograms.pdf")

@df start_end_locs histogram(:start_theta, 
    bins = 0:1.5:90,
    alpha = 0.5,
    normalize = true; 
    label="Starting Angle from mouth", 
    xlabel="Angle (radians)", 
    ylabel="Count")



@df df histogram!(:theta, 
    bins = 0:1.5:90,
    legend=true,
    alpha = 0.5,
    normalize =true,
    label="Angle from mouth", 
    xlabel="Angle (radians)", 
    ylabel="Count") 

savefig("results/plots/start_angle_histograms.pdf")

# Instead of just start/end locations, create a dataframe with consecutive differences
df_sorted = sort(df, [:track_id, :t])  # assuming you have a frame column

diffs = combine(groupby(df_sorted, :track_id),
    [:x_corr, :y_corr, :r, :theta] .=> (x -> diff(x)) .=> [:dx_corr, :dy_corr, :dr, :dtheta]
)
