library(TCGAbiolinks)
library(survminer)
library(survival)
library(tidyverse)
library(DESeq2)

grid.draw.ggsurvplot <- function(x){
  survminer:::print.ggsurvplot(x, newpage = FALSE)
}
kaplan_meier_curve <- function(cancer, ith, name) {
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
  #print(clinical)
  clinical$deceased <- ifelse(clinical$vital_status == "Alive", FALSE, TRUE)
  clinical$overall_survival <- ifelse(clinical$vital_status == "Alive",
                                          clinical$days_to_last_follow_up,
                                          clinical$days_to_death)

  clinical$overall_survival <- sapply(clinical$overall_survival, function(x) { as.integer(x/30) } )
  data <- t(data)
  data <- cbind.data.frame(rownames(data), as.numeric(data[, 1]))
  rownames(data) <- 1:nrow(data)
  colnames(data) <- c("submitter_id", "ith")
  clinical <- merge(clinical, data, by = "submitter_id", all.x = TRUE)
  clinical$ith <- ifelse(clinical$ith > median, 1, 2)
  fit <- surv_fit(Surv(overall_survival, deceased) ~ ith, data = clinical)
  res <- surv_pvalue(fit, data = clinical, method="survdiff")
  survp <- ggsurvplot(fit,
            pval = res$pval, conf.int = FALSE,
            risk.table = TRUE,
            risk.table.col = "strata",
            linetype = "strata",
            palette = c("#e77700", "#06b0f3"))
  ggsave(paste(paste("./t1/", name, sep=""), ".png", sep=""), plot = survp, 
        dpi=300, width = 10, height = 7, units = "in")
  res$pval
}


test <- function(){
  #kaplan_meier_curve("LUAD", "./subset_luad.csv")
  dir_path <- "./full/"

  # Get a list of all files in the directory
  file_list <- list.files(dir_path)
  #file_list <- head(file_list, 5)
  v1 <- c()
  v2 <- c()
  # Iterate over each file in the list
  for (file in file_list) {
    if(!grepl("unfiltered", file)){
      i <- unlist(gregexpr('_', file))[1]-1
      cancer <- toupper(substring(file,1,i))
      file_path <- paste(dir_path, file, sep="")
      v1 <- append(v1, substr(file, 1, nchar(file)-4))
      p <- kaplan_meier_curve(cancer, file_path, substr(file, 1, nchar(file)-4))
      v2 <- append(v2,p)
    }
  }
  df <- data.frame(Name = v1, Pval = v2)
  print(df)
  write.csv(df, "./r.csv")
}
#kaplan_meier_curve("BRCA", "./brca_full.csv", "full_depth2_brca2.csv")
test()