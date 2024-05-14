import pandas as pd

# read in the text file and create a dataframe
df = pd.read_csv('/home/mark_bromell/Human/appris_data_human.principal.txt', sep='\t')

# filter out rows with ALTERNATIVE:1 or ALTERNATIVE:2 in the 'APPRIS Annotation' column
df = df[~df['APPRIS Annotation'].isin(['ALTERNATIVE:1', 'ALTERNATIVE:2'])]

# create a dictionary to map principality values to numerical values
principality_map = {'PRINCIPAL:1': 1, 'PRINCIPAL:2': 2, 'PRINCIPAL:3': 3,
                    'PRINCIPAL:4': 4, 'PRINCIPAL:5': 5, 'ALTERNATIVE:1': 6,
                    'ALTERNATIVE:2': 7}

# add a new column to the dataframe with the numerical principality values
df['Principality Value'] = df['APPRIS Annotation'].map(principality_map)

# sort the dataframe by 'Gene name' and then by 'Principality Value' in ascending order
df = df.sort_values(['Gene name', 'Principality Value'])

# drop duplicates
df = df.drop_duplicates(subset='Gene name', keep='first')

# save the result to a new dataframe
result = df[['Gene name', 'Gene ID', 'Transcript ID', 'CCDS ID', 'APPRIS Annotation']]

# taking the human genePred, and filtering it such that only rows that contain the principal isoforms from above remain
human_genepred_df = pd.read_table('Homo_sapiens.GRCh38.108.genePred', sep='\t', header=None)

# below takes human_genepred_df and removes any rows that do not contain anything in appris_data_df3['Transcript ID']
human_genepred_df2 = human_genepred_df[human_genepred_df[0].isin(result['Transcript ID'])]

# dropping all rows from human_genepred_df2 if column[2] is '-'
# This would be '+' if we wanted to keep the negative strand transcripts
human_genepred_df3 = human_genepred_df2.copy()
human_genepred_df3 = human_genepred_df2.drop(human_genepred_df2[human_genepred_df2[2] == '-'].index)

# we save this and then convert this to a bed file using genePredToBed
human_genepred_df3.to_csv("/path/to/genepred_file.genePred", index=False, header=0, sep="\t")

# Read in BED file
human_bed_df = pd.read_table('/path/to/bed_file.bed', sep='\t', names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

def add_chr(x):
    return 'chr' + str(x)

# adding 'chr' to chromsome numbers
human_bed_df4 = human_bed_df2.copy()
human_bed_df4[0] = human_bed_df2[0].apply(add_chr)

# Filtering the bed file
valid_chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 
                     'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 
                     'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']
mask = human_bed_df4[0].isin(valid_chromosomes)
human_bed_df4 = human_bed_df4.loc[mask]

# saving the final BED file
human_bed_df4.to_csv("/path/to/final_bed_file.bed", index=False, header=None, sep="\t")

