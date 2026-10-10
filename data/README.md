# Dataset instructions

The project currently does not contain the NCANDA data. The original repository states that the data are not included with the code and requires contacting the authors about access and written consent.

Do not commit restricted participant-level data, demographic CSVs, fMRI time courses, or other identifiable/sensitive data to GitHub. Store approved data locally in an access-controlled location and follow the data-use agreement.

Expected inputs for `src/extract_features.py`:
- a directory of one participant's normalized time-course matrix per `.txt` file; each matrix is time points × brain regions (the research scripts use the first 269 time points);
- a demographic CSV with columns `participant_id,sex,age,scanner,heavy_drinker`.

These column names are an interface for this project, not a claim that the source file already uses them. Map the approved dataset's actual schema explicitly after checking its data dictionary. Filename stems must match `participant_id` exactly. Confirm label definitions and subject/visit uniqueness before use.

No synthetic data are to be used as scientific results. Artificial arrays may be used only for software unit tests.
