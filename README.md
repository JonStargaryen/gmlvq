# Generalized Matrix Learning Vector Quantization
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1326272.svg)](https://doi.org/10.5281/zenodo.1326272)

A Java implementation of the generalized matrix learning vector quantization, a prototype-based, 
supervised learning technique.

The project is a plug-in for the [weka machine learning framework](http://www.cs.waikato.ac.nz/ml/weka/).

## Quick start

Using the weka package manager:

- download the [leatest WEKA version](https://www.cs.waikato.ac.nz/ml/weka/downloading.html)
- download the [GMLVQ plugin zip](https://zenodo.org/record/1326272/files/gmlvq-v0.1.0-weka-plug-in.zip?download=1)
- install and run the weka gui
- choose `Tools` - `Package manager`
- in the new window, click the `File/URL` button and locate the packaged GMLVQ downloaded before
- restart WEKA

To run an analysis with GMLVQ go to the `Explorer`, choose your data and after selecting the `Classify` tab you
are able to `choose` GMLVQ located in the `functions` folder.

## Implementation Details

Generalized **Matrix** Learning Vector Quantization

Conventional LVQ was enhanced by a linear mapping rule described by an `OmegaMatrix` - putting the M
in (G**M**LVQ). This matrix has a dimension of `dataDimension x omegaDimension`. The omega dimension 
can be set to `2...dataDimension`. Depending on the chosen omega dimension each data point and 
prototype will be mapped (respectively linearly transformed) to an embedded data space. Within this 
data space distance between data points and prototypes are computed and this information is used to 
compose the update for each learning epoch. Setting the omega dimension to values significantly smaller
 than the data dimension will drastically speed up the learning process. As mapping to the embedded 
 space of data points is still computationally expensive, we 'cache' these mappings. By invoking
`DataPoint#getEmbeddedSpaceVector(OmegaMatrix)` one can retrieve the `EmbeddedSpaceVector` for this 
data point according to the specified mapping rule (provided by the `OmegaMatrix`). Results are 
directly link to the data points. So they are only calculated when absolutely necessary and previous
results can be recalled at any point. Subsequently, by calling 
`EmbeddedSpaceVector#getWinningInformation(List)` one can access the `WinningInformation` linked to 
each embedded space vector. These information include the distance to the closest prototype of the 
same class as the considered data point as well as the distance to the closest prototype of a 
different class. This information is crucial in composing the update of each epoch as well as for the 
computation of `CostFunction`s.

**Generalized** Matrix Learning Vector Quantization

Also **G**MLVQ is capable of generalization, meaning various `CostFunction` rules can be used to guide the
learning process. Most notably, it is possible to evaluate the success of each epoch by consulting 
the F-measure or precision-recall values which is especially important for problems with unbalanced
class distributions or for use cases where certain incorrect classifications (e.g. false-negatives)
could be critical.

**Visualization**

Another key feature is the possibility of tracking the influence of individual features within the 
input data which contribute the most to the training process. This is realized by a lambda matrix 
(defined as `lambda = omega * omega'`). This matrix can be visualized and will contain the influence
 of features to the classification on its principal axis. Other elements describe the correlation 
 between the corresponding features.
  
 ## Literature & References
 
 Coming soon!
