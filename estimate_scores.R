kirp_folder = "/Users/sebert/bioinformatics/final_project/data/kirp/"

sample_sheet = "gdc_sample_sheet.2023-04-11.tsv"

data_subdir = "gdc_download_20230411_043336.940718"

sample_sheet <- read.table("/Users/sebert/bioinformatics/final_project/data/kirp/gdc_sample_sheet.2023-04-11.tsv", sep = "\t", header = TRUE)


tumor_df = NULL
normal_df = NULL

library(estimate)
library(dplyr)


# for each tumor:
for (row in 1:nrow(sample_sheet)) {
	type = sample_sheet[row, "Sample.Type"]
	idd = sample_sheet[row, "Sample.ID"]
	caseid = sample_sheet[row, "Case.ID"]
	filename = sample_sheet[row, "File.Name"]
	dir = sample_sheet[row, "File.ID"]
	
		
	filepath = paste(kirp_folder, data_subdir, "/", dir, "/", filename, sep="")
	df = read.table(filepath, sep="\t", header=TRUE)
	
	
	df_filtered <- df[df$gene_type == "protein_coding", ]

	
	df_2 = data.frame(gene_name = df_filtered$gene_name, tpm_unstranded=df_filtered$tpm_unstranded)
	
	df_2 <- df_2[!is.na(df_2$tpm_unstranded),]

	df_2 <- df_2 %>% rename_with(~ idd, .cols = "tpm_unstranded")	
	
	df_2 <- distinct(df_2, gene_name, .keep_all = TRUE)


	if (type == "Solid Tissue Normal") {
		if (is.null(normal_df)) {
			normal_df = df_2
		} else {
			normal_df <- merge(normal_df, df_2, by = "gene_name", all.x = TRUE)
		}
	} else if (type == "Primary Tumor") {
		if (is.null(tumor_df)) {
			tumor_df = df_2
		} else {
			tumor_df <- merge(tumor_df, df_2, by = "gene_name", all.x = TRUE)
		}
	}	
}

fl = filter_common_genes(tumor_df, id="hgnc_symbol", tidy=TRUE)

scores = estimate_score(fl, is_affymetrix=TRUE)
	
normal_df = df <- normal_df[!(normal_df$gene_id %in% ignored_cols), ]
tumor_df = df <- tumor_df[!(tumor_df$gene_id %in% ignored_cols), ]


tumor_depth_result = DEPTH2(subset(tumor_df, select = -c(gene_id)))
normal_depth_result = DEPTH2(subset(normal_df, select = -c(gene_id)))



	
 


