import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from keras.layers import Dense, Dropout, LeakyReLU
import numpy as np

import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#os.environ['OPENBLAS_NUM_THREADS'] = '4'
def depth2(data):
    #input is a pandas data frame with gene expression values
    dt = data.T
    result = dt.apply(lambda col: (col-col.mean())/col.std() if col.std() != 0 else 0)
    result = result.replace(np.nan,0)
    result = result.apply(lambda col: abs(col)).T
    return result.std()

def get_expression_matrix(sample_sheet, data_type, data_directory):
    overall_df = None
    for i, row in sample_sheet.iterrows():
        sample_subdir = row["File ID"]
        sample_filename = row["File Name"]
        patient_id = row["Sample ID"]

        sample_type = row["Sample Type"]

        if sample_type not in ['Primary Tumor', 'Solid Tissue Normal']:
            continue

        filename = f"{data_directory}/{sample_subdir}/{sample_filename}"

        sample_df = pd.read_csv(filename, sep='\t',skiprows=[0])
               
        sample_df = sample_df[ (sample_df['gene_type'] == 'protein_coding')] 

        sample_df = sample_df[["gene_name", data_type]]
        
        sample_df = sample_df.append({'gene_name': 'sample_type', data_type: sample_type}, ignore_index=True)

        if overall_df is None:
            overall_df = sample_df[["gene_name"]]

        sample_df = sample_df[[data_type]]
    
        sample_df.columns = [patient_id]

        duplicated_cols = set(overall_df.columns) & set(sample_df.columns)

        sample_df = sample_df.drop(columns=duplicated_cols)

        overall_df = overall_df.join(sample_df) # TODO investigate duplicate sample IDs
        
    return overall_df

def get_expression_matrices(sample_sheet_path, expression_directory):
    sample_sheet = pd.read_csv(sample_sheet_path, sep="\t") 

    # sample_sheet_tumors = sample_sheet[sample_sheet["Sample Type"] == "Primary Tumor"]
    # sample_sheet_normals = sample_sheet[sample_sheet["Sample Type"] == "Solid Tissue Normal"]

    tpm = get_expression_matrix(sample_sheet, "tpm_unstranded", expression_directory)
    # normal_tpm = get_expression_matrix(sample_sheet_normals, "tpm_unstranded", expression_directory)

    fpkm = get_expression_matrix(sample_sheet, "fpkm_unstranded", expression_directory)
    # normal_fpkm = get_expression_matrix(sample_sheet_normals, "fpkm_unstranded", expression_directory)

    fpkm_uq = get_expression_matrix(sample_sheet, "fpkm_uq_unstranded", expression_directory)
    # normal_fpkm_uq = get_expression_matrix(sample_sheet_normals, "fpkm_uq_unstranded", expression_directory)

    return (tpm, fpkm, fpkm_uq)

# def survival_subset(gene_expression, important_genes, file_name):
#     copied_expression = gene_expression.copy()
#     reduced = copied_expression.loc[:, ~(gene_expression == 'Solid Tissue Normal').any()]
#     reduced = reduced.drop([19962])
#     reduced.set_index('gene_name', inplace=True)
#     subset_df = reduced[reduced.index.isin(important_genes)]
#     gene_save = depth2(subset_df).to_frame().T
#     gene_save.to_csv(file_name, index=False)

#### Below is the variable code by type:


## BEGIN KIRC
print("Starting KIRC\ns")
cancer = "kirc"

sample_sheet = "/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/data/kirc/gdc_sample_sheet.2023-04-24.tsv"
data_path = "/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/data/kirc/"

data = get_expression_matrices(sample_sheet, data_path)

fpkm = data[1]

sample_types = fpkm.iloc[-1]

fpkm = fpkm.drop(fpkm.index[-1]) # remove sample types row

gene_names = fpkm.loc[:, "gene_name"]

fpkm = fpkm.drop(labels=["gene_name"], axis=1) # remove gene names col

# min-max normalize
fpkm = fpkm.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0  else 0, axis=1)

# calculate depth scores based on full data (to use as labels for the supervised autoencoder)
fpkm_depths = depth2(fpkm)

fpkm_with_depth = fpkm.copy()

# add depth scores to the expression df
fpkm_with_depth.loc[fpkm_with_depth.index[-1] + 1] = fpkm_depths

gene_names.loc[len(gene_names)] = "depth"

# add back the gene names col
fpkm_with_depth["gene_name"] = gene_names

# add back the sample types row
fpkm_with_depth.loc[len(fpkm_with_depth)] = sample_types

print("a\n\n\n")

# remove gene names (TODO consider cleaning up/simplifying this later)
gene_names = fpkm_with_depth["gene_name"]
fpkm_with_depth = fpkm_with_depth.drop("gene_name", axis=1)

# split into 10 folds for cross-validation, and train/test both autoencoders on each fold 
# todo: decide what we need to record here (maybe we just need to save the selected genes and their train/test loss in each fold? then run depth on them later with those genes and compare to the preivous depth?)


data_and_depth = fpkm_with_depth.iloc[:-1,:] # the expressions, and the "real" depth score for each
tumor_types = fpkm_with_depth.iloc[-1,:] # tumor types (either normal or tumor; for the stratified CV)

data_and_depth = data_and_depth.T # transpose since sklearn expects rows as samples and columns as features

data_and_depth_2 = fpkm_with_depth.copy()


print("b\n\n\n\n")

skf = StratifiedKFold(n_splits=3)
splits = skf.split(data_and_depth, tumor_types)

folds = []

for _, (train_indices, test_indices) in enumerate(splits):
    # todo: decide if copying is actually necessary, possibly not 
    copied_data = data_and_depth.copy()

    train_set = copied_data.iloc[train_indices]
    test_set = copied_data.iloc[test_indices]

    full_df = fpkm_with_depth.T.copy()

    train_set_2 = full_df.iloc[train_indices].T
    test_set_2 = full_df.iloc[test_indices].T

    folds.append((train_set, test_set, train_indices, test_indices, train_set_2, test_set_2)) # todo: we may not need the indices, but they may be useful for logging

print("c\n\n\n\n")

gene_counts_to_test = [500]

gene_names_list = list(gene_names)

fold_count = 0
for fold in folds:
    full_data_train = fold[4].copy()
    full_data_test = fold[5].copy()

    train_set = fold[0]
    test_set = fold[1]

    train_set_ix = train_set.index
    test_set_ix = test_set.index

    # the depth scores, used as labels to the supervised concrete autoencoder
    train_labels = train_set.iloc[:, -1]
    test_labels = test_set.iloc[:, -1]

    # remove the depth scores from the data
    train_set = train_set.drop(train_set.columns[-1], axis=1)
    test_set = test_set.drop(test_set.columns[-1], axis=1)

    # convert to the expected data type (ndarray)
    train_labels = np.asarray(train_labels).astype('float32')
    test_labels = np.asarray(test_labels).astype('float32')

    train_set = np.asarray(train_set).astype('float32')
    test_set = np.asarray(test_set).astype('float32')

    for gene_count in gene_counts_to_test:
        # reconstruct the gene expression
        def unsupervised_output(x):
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(19962)(x) # predict all genes
            return x

        # reconstruct only the depth score from the full set
        def supervised_output(x):
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(1)(x) #predict only the depth score
            return x

        print("d\n\n\n")
	
        # TODO more reasonable epoch counts
        selector_unsupervised = ConcreteAutoencoderFeatureSelector(K = gene_count, output_function = unsupervised_output, num_epochs = 3000, learning_rate=0.002, start_temp=10, min_temp=0.1, tryout_limit=1)
        selector_supervised = ConcreteAutoencoderFeatureSelector(K = gene_count, output_function = supervised_output, num_epochs = 3000, learning_rate=0.002, start_temp=10, min_temp=0.1, tryout_limit=1)

        # the logging "should" give us everything we need

        print(f"Training supervised selector on fold:{fold_count} for {gene_count} genes")
        
        print("e\n\n\n\n")
        selector_supervised.fit(train_set, train_labels, test_set, test_labels)


        print("f\n\n\n\n")
        selected_indices = selector_supervised.get_indices()

        selected_gene_names = list()
        for index in selected_indices:
            selected_gene_names.append(gene_names_list[index])

        model = selector_supervised.get_params()

        final_test_loss = model.evaluate(test_set, test_labels)
        final_train_loss = model.evaluate(train_set, train_labels)

        output_file_name = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_supervised_{gene_count}_genes_fold_{fold_count}.txt"
        csv_output_file_name_train = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_supervised_{gene_count}_sub_depth_fold_{fold_count}_train.csv"
        csv_output_file_name_test = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_supervised_{gene_count}_sub_depth_fold_{fold_count}_test.csv"

        with open(output_file_name, "w+") as supervised_output_file:
            supervised_output_file.write("patient ids (train):\n")
            for patient_id in train_set_ix:
                supervised_output_file.write(f"{patient_id},")
            supervised_output_file.write("\n")

            supervised_output_file.write("patient ids (test):\n")
            for patient_id in test_set_ix:
                supervised_output_file.write(f"{patient_id},")
            supervised_output_file.write("\n")


            supervised_output_file.write("selected gene names:\n")
            supervised_output_file.write(f"{selected_gene_names}\n")

            supervised_output_file.write("selected indices:\n")
            supervised_output_file.write(f"{selected_indices}\n")

            supervised_output_file.write("final train loss:\n")
            supervised_output_file.write(f"{final_train_loss}\n")

            supervised_output_file.write("final test loss:\n")
            supervised_output_file.write(f"{final_test_loss}\n")

        # survival_subset(full_data_train, selected_gene_names, csv_output_file_name_train)
        # survival_subset(full_data_test, selected_gene_names, csv_output_file_name_test)

        print(f"Finished training supervised selector on fold:{fold_count} for {gene_count} genes")

        print(f"Training unsupervised selector on fold: {fold_count} for {gene_count} genes")
        selector_unsupervised.fit(train_set, train_set, test_set, test_set)

        selected_indices = selector_unsupervised.get_indices()

        selected_gene_names = list()
        for index in selected_indices:
            selected_gene_names.append(gene_names_list[index])

        # write to a file
        # * selected gene names
        # * selected indices
        # * final train loss
        # * final test loss
        # * sample ids

        # then we can do any analysis we like later with this data

        model = selector_unsupervised.get_params()

        final_test_loss = model.evaluate(test_set, test_set)
        final_train_loss = model.evaluate(train_set, train_set)

        output_file_name = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_unsupervised_{gene_count}_genes_fold_{fold_count}.txt"
        csv_output_file_name_train = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_unsupervised_{gene_count}_sub_depth_fold_{fold_count}_train.csv"
        csv_output_file_name_test = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_unsupervised_{gene_count}_sub_depth_fold_{fold_count}_test.csv"


        with open(output_file_name, "w+") as unsupervised_output_file:
            for patient_id in train_set_ix:
                unsupervised_output_file.write(f"{patient_id},")
            unsupervised_output_file.write("\n")

            unsupervised_output_file.write("patient ids (test):\n")
            for patient_id in test_set_ix:
                unsupervised_output_file.write(f"{patient_id},")
            unsupervised_output_file.write("\n")

            unsupervised_output_file.write("selected gene names:\n")
            unsupervised_output_file.write(f"{selected_gene_names}\n")

            unsupervised_output_file.write("selected indices:\n")
            unsupervised_output_file.write(f"{selected_indices}\n")

            unsupervised_output_file.write("final train loss:\n")
            unsupervised_output_file.write(f"{final_train_loss}\n")

            unsupervised_output_file.write("final test loss:\n")
            unsupervised_output_file.write(f"{final_test_loss}\n")

        # survival_subset(full_data_train, selected_gene_names, csv_output_file_name_train)
        # survival_subset(full_data_test, selected_gene_names, csv_output_file_name_test)    
        print(f"Finished training unsupervised selector on fold:{fold_count} for {gene_count} genes")
        fold_count += 1

print("finished KIRC\n")
## end KIRC



## BEGIN ACC
print("Starting ACC\ns")
cancer = "acc"

sample_sheet = "/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/data/acc/gdc_sample_sheet.2023-04-10.tsv"
data_path = "/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/data/acc/"

data = get_expression_matrices(sample_sheet, data_path)

fpkm = data[1]

sample_types = fpkm.iloc[-1]

fpkm = fpkm.drop(fpkm.index[-1]) # remove sample types row

gene_names = fpkm.loc[:, "gene_name"]

fpkm = fpkm.drop(labels=["gene_name"], axis=1) # remove gene names col

# min-max normalize
fpkm = fpkm.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0  else 0, axis=1)

# calculate depth scores based on full data (to use as labels for the supervised autoencoder)
fpkm_depths = depth2(fpkm)

fpkm_with_depth = fpkm.copy()

# add depth scores to the expression df
fpkm_with_depth.loc[fpkm_with_depth.index[-1] + 1] = fpkm_depths

gene_names.loc[len(gene_names)] = "depth"

# add back the gene names col
fpkm_with_depth["gene_name"] = gene_names

# add back the sample types row
fpkm_with_depth.loc[len(fpkm_with_depth)] = sample_types

print("a\n\n\n")

# remove gene names (TODO consider cleaning up/simplifying this later)
gene_names = fpkm_with_depth["gene_name"]
fpkm_with_depth = fpkm_with_depth.drop("gene_name", axis=1)

# split into 10 folds for cross-validation, and train/test both autoencoders on each fold 
# todo: decide what we need to record here (maybe we just need to save the selected genes and their train/test loss in each fold? then run depth on them later with those genes and compare to the preivous depth?)


data_and_depth = fpkm_with_depth.iloc[:-1,:] # the expressions, and the "real" depth score for each
tumor_types = fpkm_with_depth.iloc[-1,:] # tumor types (either normal or tumor; for the stratified CV)

data_and_depth = data_and_depth.T # transpose since sklearn expects rows as samples and columns as features

data_and_depth_2 = fpkm_with_depth.copy()


print("b\n\n\n\n")

skf = StratifiedKFold(n_splits=3)
splits = skf.split(data_and_depth, tumor_types)

folds = []

for _, (train_indices, test_indices) in enumerate(splits):
    # todo: decide if copying is actually necessary, possibly not 
    copied_data = data_and_depth.copy()

    train_set = copied_data.iloc[train_indices]
    test_set = copied_data.iloc[test_indices]

    full_df = fpkm_with_depth.T.copy()

    train_set_2 = full_df.iloc[train_indices].T
    test_set_2 = full_df.iloc[test_indices].T

    folds.append((train_set, test_set, train_indices, test_indices, train_set_2, test_set_2)) # todo: we may not need the indices, but they may be useful for logging

print("c\n\n\n\n")

gene_counts_to_test = [500]

gene_names_list = list(gene_names)

fold_count = 0
for fold in folds:
    full_data_train = fold[4].copy()
    full_data_test = fold[5].copy()

    train_set = fold[0]
    test_set = fold[1]

    train_set_ix = train_set.index
    test_set_ix = test_set.index

    # the depth scores, used as labels to the supervised concrete autoencoder
    train_labels = train_set.iloc[:, -1]
    test_labels = test_set.iloc[:, -1]

    # remove the depth scores from the data
    train_set = train_set.drop(train_set.columns[-1], axis=1)
    test_set = test_set.drop(test_set.columns[-1], axis=1)

    # convert to the expected data type (ndarray)
    train_labels = np.asarray(train_labels).astype('float32')
    test_labels = np.asarray(test_labels).astype('float32')

    train_set = np.asarray(train_set).astype('float32')
    test_set = np.asarray(test_set).astype('float32')

    for gene_count in gene_counts_to_test:
        # reconstruct the gene expression
        def unsupervised_output(x):
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(19962)(x) # predict all genes
            return x

        # reconstruct only the depth score from the full set
        def supervised_output(x):
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(1)(x) #predict only the depth score
            return x

        print("d\n\n\n")
	
        # TODO more reasonable epoch counts
        selector_unsupervised = ConcreteAutoencoderFeatureSelector(K = gene_count, output_function = unsupervised_output, num_epochs = 3000, learning_rate=0.002, start_temp=10, min_temp=0.1, tryout_limit=1)
        selector_supervised = ConcreteAutoencoderFeatureSelector(K = gene_count, output_function = supervised_output, num_epochs = 3000, learning_rate=0.002, start_temp=10, min_temp=0.1, tryout_limit=1)

        # the logging "should" give us everything we need

        print(f"Training supervised selector on fold:{fold_count} for {gene_count} genes")
        
        print("e\n\n\n\n")
        selector_supervised.fit(train_set, train_labels, test_set, test_labels)


        print("f\n\n\n\n")
        selected_indices = selector_supervised.get_indices()

        selected_gene_names = list()
        for index in selected_indices:
            selected_gene_names.append(gene_names_list[index])

        model = selector_supervised.get_params()

        final_test_loss = model.evaluate(test_set, test_labels)
        final_train_loss = model.evaluate(train_set, train_labels)

        output_file_name = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_supervised_{gene_count}_genes_fold_{fold_count}.txt"
        csv_output_file_name_train = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_supervised_{gene_count}_sub_depth_fold_{fold_count}_train.csv"
        csv_output_file_name_test = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_supervised_{gene_count}_sub_depth_fold_{fold_count}_test.csv"

        with open(output_file_name, "w+") as supervised_output_file:
            supervised_output_file.write("patient ids (train):\n")
            for patient_id in train_set_ix:
                supervised_output_file.write(f"{patient_id},")
            supervised_output_file.write("\n")

            supervised_output_file.write("patient ids (test):\n")
            for patient_id in test_set_ix:
                supervised_output_file.write(f"{patient_id},")
            supervised_output_file.write("\n")


            supervised_output_file.write("selected gene names:\n")
            supervised_output_file.write(f"{selected_gene_names}\n")

            supervised_output_file.write("selected indices:\n")
            supervised_output_file.write(f"{selected_indices}\n")

            supervised_output_file.write("final train loss:\n")
            supervised_output_file.write(f"{final_train_loss}\n")

            supervised_output_file.write("final test loss:\n")
            supervised_output_file.write(f"{final_test_loss}\n")

        # survival_subset(full_data_train, selected_gene_names, csv_output_file_name_train)
        # survival_subset(full_data_test, selected_gene_names, csv_output_file_name_test)

        print(f"Finished training supervised selector on fold:{fold_count} for {gene_count} genes")

        print(f"Training unsupervised selector on fold: {fold_count} for {gene_count} genes")
        selector_unsupervised.fit(train_set, train_set, test_set, test_set)

        selected_indices = selector_unsupervised.get_indices()

        selected_gene_names = list()
        for index in selected_indices:
            selected_gene_names.append(gene_names_list[index])

        # write to a file
        # * selected gene names
        # * selected indices
        # * final train loss
        # * final test loss
        # * sample ids

        # then we can do any analysis we like later with this data

        model = selector_unsupervised.get_params()

        final_test_loss = model.evaluate(test_set, test_set)
        final_train_loss = model.evaluate(train_set, train_set)

        output_file_name = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_unsupervised_{gene_count}_genes_fold_{fold_count}.txt"
        csv_output_file_name_train = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_unsupervised_{gene_count}_sub_depth_fold_{fold_count}_train.csv"
        csv_output_file_name_test = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_unsupervised_{gene_count}_sub_depth_fold_{fold_count}_test.csv"


        with open(output_file_name, "w+") as unsupervised_output_file:
            for patient_id in train_set_ix:
                unsupervised_output_file.write(f"{patient_id},")
            unsupervised_output_file.write("\n")

            unsupervised_output_file.write("patient ids (test):\n")
            for patient_id in test_set_ix:
                unsupervised_output_file.write(f"{patient_id},")
            unsupervised_output_file.write("\n")

            unsupervised_output_file.write("selected gene names:\n")
            unsupervised_output_file.write(f"{selected_gene_names}\n")

            unsupervised_output_file.write("selected indices:\n")
            unsupervised_output_file.write(f"{selected_indices}\n")

            unsupervised_output_file.write("final train loss:\n")
            unsupervised_output_file.write(f"{final_train_loss}\n")

            unsupervised_output_file.write("final test loss:\n")
            unsupervised_output_file.write(f"{final_test_loss}\n")

        # survival_subset(full_data_train, selected_gene_names, csv_output_file_name_train)
        # survival_subset(full_data_test, selected_gene_names, csv_output_file_name_test)    
        print(f"Finished training unsupervised selector on fold:{fold_count} for {gene_count} genes")
        fold_count += 1

print("finished ACC\n")
## end ACC




## BEGIN BRCA
print("Starting BRCA\ns")
cancer = "brca"

sample_sheet = "/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/data/brca/gdc_sample_sheet.2023-04-24.tsv"
data_path = "/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/data/brca/"

data = get_expression_matrices(sample_sheet, data_path)

fpkm = data[1]

sample_types = fpkm.iloc[-1]

fpkm = fpkm.drop(fpkm.index[-1]) # remove sample types row

gene_names = fpkm.loc[:, "gene_name"]

fpkm = fpkm.drop(labels=["gene_name"], axis=1) # remove gene names col

# min-max normalize
fpkm = fpkm.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0  else 0, axis=1)

# calculate depth scores based on full data (to use as labels for the supervised autoencoder)
fpkm_depths = depth2(fpkm)

fpkm_with_depth = fpkm.copy()

# add depth scores to the expression df
fpkm_with_depth.loc[fpkm_with_depth.index[-1] + 1] = fpkm_depths

gene_names.loc[len(gene_names)] = "depth"

# add back the gene names col
fpkm_with_depth["gene_name"] = gene_names

# add back the sample types row
fpkm_with_depth.loc[len(fpkm_with_depth)] = sample_types

print("a\n\n\n")

# remove gene names (TODO consider cleaning up/simplifying this later)
gene_names = fpkm_with_depth["gene_name"]
fpkm_with_depth = fpkm_with_depth.drop("gene_name", axis=1)

# split into 10 folds for cross-validation, and train/test both autoencoders on each fold 
# todo: decide what we need to record here (maybe we just need to save the selected genes and their train/test loss in each fold? then run depth on them later with those genes and compare to the preivous depth?)


data_and_depth = fpkm_with_depth.iloc[:-1,:] # the expressions, and the "real" depth score for each
tumor_types = fpkm_with_depth.iloc[-1,:] # tumor types (either normal or tumor; for the stratified CV)

data_and_depth = data_and_depth.T # transpose since sklearn expects rows as samples and columns as features

data_and_depth_2 = fpkm_with_depth.copy()


print("b\n\n\n\n")

skf = StratifiedKFold(n_splits=3)
splits = skf.split(data_and_depth, tumor_types)

folds = []

for _, (train_indices, test_indices) in enumerate(splits):
    # todo: decide if copying is actually necessary, possibly not 
    copied_data = data_and_depth.copy()

    train_set = copied_data.iloc[train_indices]
    test_set = copied_data.iloc[test_indices]

    full_df = fpkm_with_depth.T.copy()

    train_set_2 = full_df.iloc[train_indices].T
    test_set_2 = full_df.iloc[test_indices].T

    folds.append((train_set, test_set, train_indices, test_indices, train_set_2, test_set_2)) # todo: we may not need the indices, but they may be useful for logging

print("c\n\n\n\n")

gene_counts_to_test = [500]

gene_names_list = list(gene_names)

fold_count = 0
for fold in folds:
    full_data_train = fold[4].copy()
    full_data_test = fold[5].copy()

    train_set = fold[0]
    test_set = fold[1]

    train_set_ix = train_set.index
    test_set_ix = test_set.index

    # the depth scores, used as labels to the supervised concrete autoencoder
    train_labels = train_set.iloc[:, -1]
    test_labels = test_set.iloc[:, -1]

    # remove the depth scores from the data
    train_set = train_set.drop(train_set.columns[-1], axis=1)
    test_set = test_set.drop(test_set.columns[-1], axis=1)

    # convert to the expected data type (ndarray)
    train_labels = np.asarray(train_labels).astype('float32')
    test_labels = np.asarray(test_labels).astype('float32')

    train_set = np.asarray(train_set).astype('float32')
    test_set = np.asarray(test_set).astype('float32')

    for gene_count in gene_counts_to_test:
        # reconstruct the gene expression
        def unsupervised_output(x):
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(19962)(x) # predict all genes
            return x

        # reconstruct only the depth score from the full set
        def supervised_output(x):
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(600)(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.1)(x)
            x = Dense(1)(x) #predict only the depth score
            return x

        print("d\n\n\n")
	
        # TODO more reasonable epoch counts
        selector_unsupervised = ConcreteAutoencoderFeatureSelector(K = gene_count, output_function = unsupervised_output, num_epochs = 3000, learning_rate=0.002, start_temp=10, min_temp=0.1, tryout_limit=1)
        selector_supervised = ConcreteAutoencoderFeatureSelector(K = gene_count, output_function = supervised_output, num_epochs = 3000, learning_rate=0.002, start_temp=10, min_temp=0.1, tryout_limit=1)

        # the logging "should" give us everything we need

        print(f"Training supervised selector on fold:{fold_count} for {gene_count} genes")
        
        print("e\n\n\n\n")
        selector_supervised.fit(train_set, train_labels, test_set, test_labels)


        print("f\n\n\n\n")
        selected_indices = selector_supervised.get_indices()

        selected_gene_names = list()
        for index in selected_indices:
            selected_gene_names.append(gene_names_list[index])

        model = selector_supervised.get_params()

        final_test_loss = model.evaluate(test_set, test_labels)
        final_train_loss = model.evaluate(train_set, train_labels)

        output_file_name = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_supervised_{gene_count}_genes_fold_{fold_count}.txt"
        csv_output_file_name_train = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_supervised_{gene_count}_sub_depth_fold_{fold_count}_train.csv"
        csv_output_file_name_test = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_supervised_{gene_count}_sub_depth_fold_{fold_count}_test.csv"

        with open(output_file_name, "w+") as supervised_output_file:
            supervised_output_file.write("patient ids (train):\n")
            for patient_id in train_set_ix:
                supervised_output_file.write(f"{patient_id},")
            supervised_output_file.write("\n")

            supervised_output_file.write("patient ids (test):\n")
            for patient_id in test_set_ix:
                supervised_output_file.write(f"{patient_id},")
            supervised_output_file.write("\n")


            supervised_output_file.write("selected gene names:\n")
            supervised_output_file.write(f"{selected_gene_names}\n")

            supervised_output_file.write("selected indices:\n")
            supervised_output_file.write(f"{selected_indices}\n")

            supervised_output_file.write("final train loss:\n")
            supervised_output_file.write(f"{final_train_loss}\n")

            supervised_output_file.write("final test loss:\n")
            supervised_output_file.write(f"{final_test_loss}\n")

        # survival_subset(full_data_train, selected_gene_names, csv_output_file_name_train)
        # survival_subset(full_data_test, selected_gene_names, csv_output_file_name_test)

        print(f"Finished training supervised selector on fold:{fold_count} for {gene_count} genes")

        print(f"Training unsupervised selector on fold: {fold_count} for {gene_count} genes")
        selector_unsupervised.fit(train_set, train_set, test_set, test_set)

        selected_indices = selector_unsupervised.get_indices()

        selected_gene_names = list()
        for index in selected_indices:
            selected_gene_names.append(gene_names_list[index])

        # write to a file
        # * selected gene names
        # * selected indices
        # * final train loss
        # * final test loss
        # * sample ids

        # then we can do any analysis we like later with this data

        model = selector_unsupervised.get_params()

        final_test_loss = model.evaluate(test_set, test_set)
        final_train_loss = model.evaluate(train_set, train_set)

        output_file_name = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_unsupervised_{gene_count}_genes_fold_{fold_count}.txt"
        csv_output_file_name_train = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_unsupervised_{gene_count}_sub_depth_fold_{fold_count}_train.csv"
        csv_output_file_name_test = f"/a/buffalo.cs.fiu.edu./disk/jccl-001/homes/seber007/final_project/outputs/{cancer}_unsupervised_{gene_count}_sub_depth_fold_{fold_count}_test.csv"


        with open(output_file_name, "w+") as unsupervised_output_file:
            for patient_id in train_set_ix:
                unsupervised_output_file.write(f"{patient_id},")
            unsupervised_output_file.write("\n")

            unsupervised_output_file.write("patient ids (test):\n")
            for patient_id in test_set_ix:
                unsupervised_output_file.write(f"{patient_id},")
            unsupervised_output_file.write("\n")

            unsupervised_output_file.write("selected gene names:\n")
            unsupervised_output_file.write(f"{selected_gene_names}\n")

            unsupervised_output_file.write("selected indices:\n")
            unsupervised_output_file.write(f"{selected_indices}\n")

            unsupervised_output_file.write("final train loss:\n")
            unsupervised_output_file.write(f"{final_train_loss}\n")

            unsupervised_output_file.write("final test loss:\n")
            unsupervised_output_file.write(f"{final_test_loss}\n")

        # survival_subset(full_data_train, selected_gene_names, csv_output_file_name_train)
        # survival_subset(full_data_test, selected_gene_names, csv_output_file_name_test)    
        print(f"Finished training unsupervised selector on fold:{fold_count} for {gene_count} genes")
        fold_count += 1

print("finished BRCA\n")
## end BRCA

