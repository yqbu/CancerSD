# CancerSD: Cancer Subtype Diagnosis with Limited Multi-Omics Data
CancerSD is an end-to-end model designed for \underline{Cancer} \underline{S}ubtype \underline{D}iagnosis using limited weakly-paired multi-omics data

The diagnosis of cancer subtypes is a prerequisite for precise treatment. Compared to single-omics data-based diagnostic solutions, multi-omics data fusion-based approaches often provide a more accurate and authentic diagnosis. However, they build on the requisite of sufficient samples with completely-paired omics data, which is challenging to obtain in clinical applications. In this study, a novel integrative model (CancerSD) is proposed for cancer subtype diagnosis using limited samples with weakly-paired multi-omics data. CancerSD constructs tailored contrastive learning and masking-and-reconstruction tasks to effectively impute the missing omics, enabling flexible and accurate cancer subtype diagnosis. To cope with scarce clinical samples, it introduces a category-level contrastive loss and extends the meta-learning framework to mine specific knowledge from external datasets. Experiments on benchmark datasets show that CancerSD not only gives accurate diagnosis, but also maintains a high authenticity and good interpretability. CancerSD identifies important molecular characteristics associated with cancer subtypes, and it defines the integrated CancerSD score that can serve as an independent predictive factor for patient prognosis.
