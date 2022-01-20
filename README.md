# EM_OligomerAnalysis

The analysis of flexible multivalent oligomers on EM micrographs consists of the three following steps:
1. Identifying and scoring oligomers from particle-picked micrographs.
2. Distance-filtering oligomers to avoid crowded regions.
3. Statistical correction procedure to ameliorate random proximity artifacts.

The theory behind this analysis as well as detailed pseudocodes outlining all procedures, such as defining input/output files, can be found in this publication:
https://doi.org/10.1101/2020.06.16.154096   

---

### 1. Identifying and scoring oligomers from particle-picked micrographs.

Run this code in the following way:<br />
`python <score_oligomer.py>   <picks_file>  <peaks_file>   <cdfs_dir>  <score_threshold>` <br />
where<br /> 
`<picks_file>` is the file with coordinates of picked particles (DoG picker output file, ‘star’ format)<br />
`<peaks_file>` is the micrograph image file (DoG picker output file, ‘png’ format)<br />
`<cdfs_dir>`   is the path to the directory of discretized CDFs required for the scoring algorithm<br />
`<score_threshold>` is the the value below which identified oligomers are not considered.

The output is a compressed numpy array file (`.npz`) with saved arrays of particle coordinates, oligomer coordinates, and corresponding oligomer scores.

---

### 2. Distance-filtering oligomers to avoid crowded regions.

Run this code in the following way:<br />
`python distance_filtering.py   <file_in>  <min_dist>`<br />
where<br /> 
`<file_in>`  is the output file from `<score_oligomer.py>`<br />
`<min_dist>` is filtering distance for oligomers (in nm).

The output is a compressed numpy array file (`.npz`) with saved arrays of free LC8 and oligomer coordinates, that are identified to be dinstance-filtered or not.

---

### 3. Statistical correction procedure to ameliorate random proximity artifacts.

Run this code in the following way:<br />
`python statistical_correction.py   <file_in>`<br />
where<br /> 
`<file_in>`  is the output file from `<distance_filtering.py>`.





