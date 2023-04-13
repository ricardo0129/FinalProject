library(TCGAbiolinks)
library(survminer)
library(survival)
library(SummarizedExperiment)
library(tidyverse)
library(DESeq2)

clinical <- GDCquery_clinic("TCGA-ACC")
any(colnames(clinical) %in% c("submitter_id","vital_status", "days_to_last_follow_up", "days_to_death"))
r <- which(colnames(clinical) %in% c("submitter_id","vital_status", "days_to_last_follow_up", "days_to_death"))

head(clinical[,r],30)

clinical$deceased <- ifelse(clinical$vital_status == "Alive", FALSE, TRUE)

clinical$overall_survival <- ifelse(clinical$vital_status == "Alive",
                                         clinical$days_to_last_follow_up,
                                         clinical$days_to_death)

fit <- survfit(Surv(overall_survival,deceased) ~ gender, data=clinical)
ggsurvplot(fit,
           pval = TRUE, conf.int = FALSE,
           risk.table = TRUE, # Add risk table
           risk.table.col = "strata", # Change risk table color by groups
           linetype = "strata", # Change line type by groups
           ggtheme = theme_bw(), # Change ggplot2 theme
           palette = c("#E7B800", "#2E9FDF"))

