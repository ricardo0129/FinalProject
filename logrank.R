library("survival")
library("survminer")
library("Rcpp")

data("lung")
lung$test<- ifelse(lung$age<64, 1, 2)
fit <- survfit(Surv(time, status) ~ sex, data = lung)
print(lung$time)
print(lung$status)
print(fit)
summary(fit)$table
ggsurvplot(fit,
           pval = TRUE, conf.int = TRUE,
           risk.table = TRUE, # Add risk table
           risk.table.col = "strata", # Change risk table color by groups
           linetype = "strata", # Change line type by groups
           surv.median.line = "hv", # Specify median survival
           ggtheme = theme_bw(), # Change ggplot2 theme
           palette = c("#E7B800", "#2E9FDF"))
surv_diff <- survdiff(Surv(time, status) ~ sex, data = lung)
surv_diff
