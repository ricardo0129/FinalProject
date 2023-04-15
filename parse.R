library(TCGAbiolinks)
library(survminer)
library(survival)
library(tidyverse)
library(DESeq2)

kaplan_meier_curve <- function(cancer, ith) {
  #Input is the name of a Cancer eg. ACC & File with patient ids with ith values
  #Produces a kaplan meier curve saved under Rplots
  clinical <- GDCquery_clinic(paste("TCGA-", cancer, sep = ""))
  r <- which(colnames(clinical) %in% c("submitter_id","vital_status", "days_to_last_follow_up", "days_to_death"))
  clinical <- clinical[, r]
  data <- read.csv(ith, check.names = FALSE)
  col <- colnames(data)
  for (i in seq_along(col)) {
    col[i] <- substring(col[i], 1, 12)
  }
  colnames(data) <- col
  median <- apply(data, 1, median, na.rm = TRUE)
  clinical <- subset(clinical, clinical$submitter_id %in% col)

  clinical$deceased <- ifelse(clinical$vital_status == "Alive", FALSE, TRUE)
  clinical$overall_survival <- ifelse(clinical$vital_status == "Alive",
                                          clinical$days_to_last_follow_up,
                                          clinical$days_to_death)
  data <- t(data)
  data <- cbind.data.frame(rownames(data), as.numeric(data[, 1]))
  rownames(data) <- 1:nrow(data)
  colnames(data) <- c("submitter_id", "ith")
  clinical <- merge(clinical, data, by = "submitter_id", all.x = TRUE)
  clinical$ith <- ifelse(clinical$ith > median, 1, 2)
  print(head(clinical))
  fit <- surv_fit(Surv(overall_survival, deceased) ~ ith, data = clinical)
  ggsurvplot(fit,
            pval = TRUE, conf.int = FALSE,
            risk.table = TRUE,
            risk.table.col = "strata",
            linetype = "strata",
            palette = c("#e77700", "#06b0f3"))
}

kaplan_meier_curve("ACC", "./test.csv")