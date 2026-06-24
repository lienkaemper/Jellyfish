Date: Monday, May 11

What I did

1) Wrote/generated code to simulate the jellyfish equations (clean version, radially symmetric, 1d)

   - this code is in the file sim_jellyfish_eqn_clean.ipynb, which is in scripts 

2) Wrote/generated code to identify jumps -- times when the jellyfish is recovering off-scope. 
   
    - this code is in identify_jumpys.ipynb. This is working, and for each dataset, I've saved a clean version, called final_nomouth_processed.csv. 
    - this has a column real_time, which is in hours 
    - also has some convenience variabes like x_norm, y_norm (x and y coordinites with mouth at origin), r , theta (polar coords with mouth at origin) 

3) Started, but did not finish inferring parameters. 
   - this is in a notebook called infer_parameters.ipynb
   - so far, have estimated the radial density of neurons, births, and deaths for all datasets. 
   - need to start using this to estimate parameters. 

Date: Thursday, May 28

What I did:

1) Catching up to what I did in the past, understanding what the setup of the code was. 
   - It looks like "identify_jumps.nb" loops through all datasets, identifies when the jellyfish is resting off-scope, and makes a real-time column. It also comptues convenience variables like r and theta for all datasets 
   - where does the "real analysis" happen? 
     - actual analysis means inferring paramegers. 
     - "infer_params.ipynb" is also set up to loop through datasets. I'm not sure if there's anything I was historically computing that isn't represented there. 

Date: Monday, June 8

Goal: write some code to estimate how drift velocities vary spatially. 

- this is in radius_dependent_migration.ipynb
- currently just doing this based on bin initially observed in
- I want to switch to doing it based on bin at a particular timestep. 
- currently, relationship seems non-monotonic: neurons initially observed in mouth or around rim migrate the least
- 

Date: Tuesday, June 23, 2023

Goal: 
1) understand how the parameters describing the progenitor evolution (drift velocity, diffusion coefficient, birth rate, and differentiation rate) shape the distribution of progenitor births. Wrote a pipeline to 
   a) simulate the progenitor equation only, 
   b) find the total birth rate, mean birth radius, and birth radius variance, and 
   c) plot these things on a grid of values 
Docs available at scripts/progenitor_sweep_README.md