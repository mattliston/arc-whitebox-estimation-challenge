DRY RUN ONLY - NOT A CANDIDATE (trainer, 10-12 07:5xZ, for the engineer's text-only port rehearsal of K4L age 2 by layer).
params.pt = S50's frozen point (cert/RAD/S50/point_v1/params.pt) with pl['k4la2'] + 0.05 at layers 1-7 and + 0.02 at layers 8-15 (nothing else changed).
params_exportable_v1.pt = cert/RAD/make_exportable.py of it (k4la2 layer 0 <- layer 1, the usual fills).
candidate.py / physics_dials.json / per_layer_dials.json = export_dials_s46_v5.py of params_exportable_v1.pt (sidecar 'k4la2' = 0.775 at L0-7, 0.745 at L8-15;
builder_levers K4L_per_age[1] = the 16-list). head_weights_31x128.json = rows 0-30 of S50's head (unchanged; = Nd's table 03794864).
