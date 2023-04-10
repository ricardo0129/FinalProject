library(TCGAbiolinks)
library(survminer)
library(survival)
library(SummarizedExperiment)
library(tidyverse)
library(DESeq2)

# getting clinical data for TCGA-BRCA cohort -------------------
clinical_brca <- GDCquery_clinic("TCGA-BRCA")
#colnames(clinical_brca)
any(colnames(clinical_brca) %in% c("submitter_id","vital_status", "days_to_last_follow_up", "days_to_death"))
r <- which(colnames(clinical_brca) %in% c("submitter_id","vital_status", "days_to_last_follow_up", "days_to_death"))
clinical_brca[,r]


# looking at some variables associated with survival 
table(clinical_brca$vital_status)

# days_to_death, that is the number of days passed from the initial diagnosis to the patient’s death (clearly, this is only relevant for dead patients)
#days_to_last_follow_up that is the number of days passed from the initial diagnosis to the last visit.

# change certain values the way they are encoded
clinical_brca$deceased <- ifelse(clinical_brca$vital_status == "Alive", FALSE, TRUE)

# create an "overall survival" variable that is equal to days_to_death
# for dead patients, and to days_to_last_follow_up for patients who
# are still alive
clinical_brca$overall_survival <- ifelse(clinical_brca$vital_status == "Alive",
                                         clinical_brca$days_to_last_follow_up,
                                         clinical_brca$days_to_death)
