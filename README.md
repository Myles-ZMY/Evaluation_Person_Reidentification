# Evaluation_Person_Reidentification
Evaluate the performance of the CNN on the task of person re-identification.

## Introduction
Given a sequence of features as library sequence and another sequence of features as query sequence, this tool adopts XQDA as metric learning method and can find two nearest features from library sequence and query sequence respectively, then uses CMC curve to show the performance.
A sample ".mat" file is given, which is the result of [CNN-based person re-identification](https://github.com/riceroll/CNN_Person_Reidentification)

## Requirements

- Matlab-2015b

## Quick Start
### Data
A sample file "feature.mat" file is saved in "./data". You can directly run the code.

## References

- [Person Re-identification by Local Maximal Occurrence Representation and Metric Learning](http://www.cbsr.ia.ac.cn/users/scliao/projects/lomo_xqda/)
