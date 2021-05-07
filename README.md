# Algorithms for Genomics Cancer Classification

There is a growing interest for incorporating genomic data into both research and clinical workflows, such as diagnostic decision support and tailoring disease treatment and prevention strategies to individual genomic characteristics of patients. To be able to use genetic characteristics for such purposes, it is imperative to identify biomarkers, genetic features associated with disease risk, diagnostic, prognosis, or response to treatment. This is an example implementation of feature selection / identification approaches from Machine Learning (ML) that peform well on the task of biomarker identification on genomic data.

We have used some specific Amazon SageMaker features as part of the implementation, namely:
* SageMaker Processing
* SageMaker Model Training
* Hyperparameter Optimization

We leveraged two genomic datasets for this evaluation, TCGA and GTex. The Cancer Genome Atlas (TCGA) includes a large compendium of thousands of tumors with RNA-seq gene expression measurements of over 20,000 genes. The GTEx dataset consists of samples mostly from healthy subjects and has a higher amount of genomic features (34,218 features).

To start with the implementation, you will have to build the containers of individual model artefacts (to be used for training) to ECR. You can do this by running the `buidl_and_push` script of invdividual artefacts (lightgbm, autogluon, and imodels). For example, to build the `lightgbm` model, run:

`./build_and_push.sh genomics-cancer-classification-lightgbm`

Subsequently, follow the steps in the notebook