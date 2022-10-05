# Ancient-Chinese-Character-OCR
- Database from [MTHv2](https://github.com/HCIILAB/MTHv2_Datasets_Release)

## WorkFlow
![pic](https://github.com/bill10655/Ancient-Chinese-Character-OCR/blob/main/Model%20structure.png)

### Pre-processing
First, due to the data icomes from the real world, it will be damaged inevitably. Therefore, we use the possiblity to filter out those won't be able to form a word and might just be a dirty thing or other non-word things, for example, we get rid of a huge black area or a black straight line cross the whole pages. After that, we adopt DBSCAN(Density-based spatial clustering of applications with noise) algorithm to further get possible cluster candidates and the rest will be removed. Finally, we use the classic image processing methods, i.e. erosion & dilation, to avoid interfere from noises.

### Object detection model
### Language model support

## Result Visualization
![result](https://github.com/bill10655/Ancient-Chinese-Character-OCR/blob/main/OCR_results.png)

## References
- https://github.com/HCIILAB/MTHv2_Datasets_Release
