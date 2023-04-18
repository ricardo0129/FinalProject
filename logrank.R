library(CancerRxGene)
library(tidyverse)

# Define list of cell lines to download
cell_lines <- c("A549", "MCF7", "PC3")

# Download gene expression data for the specified cell lines
gene_expr <- rx_get_genomic_profiles(cell_lines, genomic_data_type = "gene_expression")

# Transpose the data and set cell line names as column names
gene_expr_t <- gene_expr %>%
  select(-c(Gene_symbol, Entrez_gene_id)) %>%
  t() %>%
  as.data.frame() %>%
  setNames(cell_lines)

# Save the data to a CSV file
write_csv(gene_expr_t, "gene_expression.csv")