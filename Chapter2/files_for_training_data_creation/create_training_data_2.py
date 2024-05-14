import pandas as pd
from itertools import zip_longest
import numpy as np
import random
import os

np.random.seed(42)
random.seed(42)

acceptor_seq_len = 210

# read in positive strand bed created from create_training_data_1.py
human_bed_princiso_noneg_multiexon_df = pd.read_csv('/path/to/final/pos_bed/from/create_training_data_1.bed', sep='\t', 
                                                    names=['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 
                                                           'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 
                                                           'blockStarts'])

# read in human_bed_princiso_nopos_multiexon.bed
human_bed_princiso_nopos_multiexon_df = pd.read_csv('/path/to/final/neg_bed/from/create_training_data_1.bed', sep='\t', 
                                                    names=['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 
                                                           'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 
                                                           'blockStarts'])


# convert blockStarts and blockSizes fom str to list of strings to list of ints
human_bed_princiso_noneg_multiexon_df['blockSizes'] = human_bed_princiso_noneg_multiexon_df.blockSizes.apply(lambda x: x[0:-1].split(','))
human_bed_princiso_noneg_multiexon_df['blockStarts'] = human_bed_princiso_noneg_multiexon_df.blockStarts.apply(lambda x: x[0:-1].split(','))
human_bed_princiso_noneg_multiexon_df['blockSizes'] = [[int(x) for x in l] for l in human_bed_princiso_noneg_multiexon_df['blockSizes']]
human_bed_princiso_noneg_multiexon_df['blockStarts'] = [[int(x) for x in l] for l in human_bed_princiso_noneg_multiexon_df['blockStarts']]

# convert blockStarts and blockSizes fom str to list of strings to list of ints
human_bed_princiso_nopos_multiexon_df['blockSizes'] = human_bed_princiso_nopos_multiexon_df.blockSizes.apply(lambda x: x[0:-1].split(','))
human_bed_princiso_nopos_multiexon_df['blockStarts'] = human_bed_princiso_nopos_multiexon_df.blockStarts.apply(lambda x: x[0:-1].split(','))
human_bed_princiso_nopos_multiexon_df['blockSizes'] = [[int(x) for x in l] for l in human_bed_princiso_nopos_multiexon_df['blockSizes']]
human_bed_princiso_nopos_multiexon_df['blockStarts'] = [[int(x) for x in l] for l in human_bed_princiso_nopos_multiexon_df['blockStarts']]

# this gives the exon start and end coordinates relative to the positive strand transcription start site
human_bed_princiso_noneg_multiexon_df2 = human_bed_princiso_noneg_multiexon_df.assign(blockEnds=[
    [x + y for x, y in zip_longest(d, c, fillvalue=0)] 
    for d, c in zip(human_bed_princiso_noneg_multiexon_df.blockStarts, human_bed_princiso_noneg_multiexon_df.blockSizes)
    ])

human_bed_princiso_nopos_multiexon_df2 = human_bed_princiso_nopos_multiexon_df.assign(blockEnds=[
    [x + y for x, y in zip_longest(d, c, fillvalue=0)] 
    for d, c in zip(human_bed_princiso_nopos_multiexon_df.blockStarts, human_bed_princiso_nopos_multiexon_df.blockSizes)
    ])

#======================================== BED1 (donors noneg) =============================================#

donor_seq_bed = human_bed_princiso_noneg_multiexon_df2.copy()

# Remove chromEnd, score, itemRgb, blockCount, blockSizes and blockStarts
donor_seq_bed.drop(donor_seq_bed.loc[:, ['chromEnd', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']], axis=1, inplace=True)

# pop blockEnds out and insert it into position 1, rename it donorSeqs
second_column = donor_seq_bed.pop('blockEnds')
donor_seq_bed.insert(2, 'donorSeqs', second_column)

# Remove last blockEnd from every list in donorSeqs
donor_seq_bed['donorSeqs'] = [l[:-1] for l in donor_seq_bed['donorSeqs']]

# Expand each list such that each element in each list gets its own row (using df.explode())
donor_seq_bed_2 = donor_seq_bed.explode('donorSeqs').reset_index(drop=True)

# Sum chromStart and donorSeq and save in new column
sum_column = donor_seq_bed_2['chromStart'] + donor_seq_bed_2['donorSeqs']
donor_seq_bed_2['donorCoords'] = sum_column

# Remove chromStart and donorSeqs from dataframe
donor_seq_bed_2.drop(donor_seq_bed_2.loc[:, ['chromStart', 'donorSeqs']], axis=1, inplace=True)

# Move donorCoords
second_column_2 = donor_seq_bed_2.pop('donorCoords')
donor_seq_bed_2.insert(2, 'donorCoords', second_column_2)

# add acceptor_seq_len nucleotides
add_asl = donor_seq_bed_2['donorCoords'] + acceptor_seq_len +2
donor_seq_bed_2['donorCoords_plus_seq_len'] = add_asl

# subtract acceptor_seq_len nucleotides
minus_asl = donor_seq_bed_2['donorCoords'] - acceptor_seq_len
donor_seq_bed_2['donorCoords_minus_seq_len'] = minus_asl

donor_seq_bed_2 = donor_seq_bed_2.drop('donorCoords', axis=1)
donor_seq_bed_2 = donor_seq_bed_2[['chrom', 'donorCoords_minus_seq_len', 'donorCoords_plus_seq_len', 'name']]

################only if necessary#################
# Add chr to start of every number 
# donor_seq_bed_2['chrom'] = 'chr' + donor_seq_bed_2['chrom'].astype(str)
##################################################

# adding suffixes Transcript ID to distinguish between exons
g = donor_seq_bed_2.groupby(['chrom', 'name'])
donor_seq_bed_2.loc[g['name'].transform('size').gt(1),
       'name'] += '_'+g.cumcount().astype(str)

# Add '_pos' to the end of every name
donor_seq_bed_2['name'] = donor_seq_bed_2['name'].astype(str) + '_pos'

# Write donor_seq_bed_2 to a new CSV file
filename1 = "/path/to/splice_site_sequences_1.bed"
donor_seq_bed_2.to_csv(filename1, index=False, sep="\t", header=None)

# Check that the file exists before printing the confirmation message
if os.path.isfile(filename1):
    print(f"Saved as {filename1}")
else:
    print(f"Failed to save {filename1}")

#======================================== BED2 (acceptors noneg)=============================================#

acceptor_seq_bed1 = human_bed_princiso_noneg_multiexon_df2.copy()

acceptor_seq_bed1.drop(acceptor_seq_bed1.loc[:, ['chromEnd', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockEnds']], axis=1, inplace=True)

second_column_31 = acceptor_seq_bed1.pop('blockStarts')

acceptor_seq_bed1.insert(2, 'acceptorSeqs', second_column_31)

acceptor_seq_bed1['acceptorSeqs'] = [l[1:] for l in acceptor_seq_bed1['acceptorSeqs']]

acceptor_seq_bed_21 = acceptor_seq_bed1.explode('acceptorSeqs').reset_index(drop=True)

sum_column = acceptor_seq_bed_21['chromStart'] + acceptor_seq_bed_21['acceptorSeqs']

acceptor_seq_bed_21['acceptorCoords'] = sum_column

acceptor_seq_bed_21.drop(acceptor_seq_bed_21.loc[:, ['chromStart', 'acceptorSeqs']], axis=1, inplace=True)

second_column_41 = acceptor_seq_bed_21.pop('acceptorCoords')

acceptor_seq_bed_21.insert(1, 'acceptorCoords', second_column_41)

subtract_seq_len_21 = acceptor_seq_bed_21['acceptorCoords'] - acceptor_seq_len -2

acceptor_seq_bed_21['acceptorCoords_minus_seq_len'] = subtract_seq_len_21

add_seq_len_21 = acceptor_seq_bed_21['acceptorCoords'] + acceptor_seq_len

acceptor_seq_bed_21['acceptorCoords_plus_seq_len'] = add_seq_len_21

acceptor_seq_bed_21 = acceptor_seq_bed_21.drop('acceptorCoords', axis=1)

acceptor_seq_bed_21 = acceptor_seq_bed_21[['chrom', 'acceptorCoords_minus_seq_len', 'acceptorCoords_plus_seq_len', 'name']]

##################if necessary##################
# Add chr to start of every number 
# acceptor_seq_bed_21['chrom'] = 'chr' + acceptor_seq_bed_21['chrom'].astype(str)
################################################

g1 = acceptor_seq_bed_21.groupby(['chrom', 'name'])
acceptor_seq_bed_21.loc[g1['name'].transform('size').gt(1),
       'name'] += '_'+g1.cumcount().astype(str)

acceptor_seq_bed_21['name'] = acceptor_seq_bed_21['name'].astype(str) + '_pos'

filename1 = "/path/to/splice_site_sequences_2.bed"
acceptor_seq_bed_21.to_csv(filename1, index=False, sep="\t", header=None)

# Check that the file exists before printing the confirmation message
if os.path.isfile(filename1):
    print(f"Saved as {filename1}")
else:
    print(f"Failed to save {filename1}")

#=================================== BED1 (acceptors (treated like donors) nopos)========================================#

acceptor_seq_bed2 = human_bed_princiso_nopos_multiexon_df2.copy()

acceptor_seq_bed2.drop(acceptor_seq_bed2.loc[:, ['chromEnd', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']], axis=1, inplace=True)

second_column2 = acceptor_seq_bed2.pop('blockEnds')

acceptor_seq_bed2.insert(2, 'acceptorSeqs', second_column2)

acceptor_seq_bed2['acceptorSeqs'] = [l[:-1] for l in acceptor_seq_bed2['acceptorSeqs']]

acceptor_seq_bed_22 = acceptor_seq_bed2.explode('acceptorSeqs').reset_index(drop=True)

sum_column2 = acceptor_seq_bed_22['chromStart'] + acceptor_seq_bed_22['acceptorSeqs']

acceptor_seq_bed_22['acceptorCoords'] = sum_column2

acceptor_seq_bed_22.drop(acceptor_seq_bed_22.loc[:, ['chromStart', 'acceptorSeqs']], axis=1, inplace=True)

second_column_22 = acceptor_seq_bed_22.pop('acceptorCoords')

acceptor_seq_bed_22.insert(2, 'acceptorCoords', second_column_22)

add_seq_len = acceptor_seq_bed_22['acceptorCoords'] + acceptor_seq_len + 2

acceptor_seq_bed_22['acceptorCoords_plus_seq_len'] = add_seq_len

minus_seq_len = acceptor_seq_bed_22['acceptorCoords'] - acceptor_seq_len

acceptor_seq_bed_22['acceptorCoords_minus_seq_len'] = minus_seq_len

acceptor_seq_bed_22 = acceptor_seq_bed_22.drop('acceptorCoords', axis=1)

acceptor_seq_bed_22 = acceptor_seq_bed_22[['chrom', 'acceptorCoords_minus_seq_len', 'acceptorCoords_plus_seq_len', 'name']]

#################if necessary##################
# Add chr to start of every number 
# acceptor_seq_bed_22['chrom'] = 'chr' + acceptor_seq_bed_22['chrom'].astype(str)
###############################################

g2 = acceptor_seq_bed_22.groupby(['chrom', 'name'])
acceptor_seq_bed_22.loc[g2['name'].transform('size').gt(1),
       'name'] += '_'+g2.cumcount().astype(str)

acceptor_seq_bed_22['name'] = acceptor_seq_bed_22['name'].astype(str) + '_neg'
print(f"acceptor neg: {len(acceptor_seq_bed_22)}")

filename2 = "/path/to/splice_site_sequences_3.bed"
acceptor_seq_bed_22.to_csv(filename2, index=False, sep="\t", header=None)

# Check that the file exists before printing the confirmation message
if os.path.isfile(filename2):
    print(f"Saved as {filename2}")
else:
    print(f"Failed to save {filename2}")

# # =================================== BED2 (donors (treated like acceptors) nopos) ========================================#

donor_seq_bed_d2 = human_bed_princiso_nopos_multiexon_df2.copy()

donor_seq_bed_d2.drop(donor_seq_bed_d2.loc[:, ['chromEnd', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockEnds']], axis=1, inplace=True)

second_column_3_d2 = donor_seq_bed_d2.pop('blockStarts')

donor_seq_bed_d2.insert(2, 'donorSeqs', second_column_3_d2)

donor_seq_bed_d2['donorSeqs'] = [l[1:] for l in donor_seq_bed_d2['donorSeqs']]

donor_seq_bed_2_d2 = donor_seq_bed_d2.explode('donorSeqs').reset_index(drop=True)

sum_column_d2 = donor_seq_bed_2_d2['chromStart'] + donor_seq_bed_2_d2['donorSeqs']

donor_seq_bed_2_d2['donorCoords'] = sum_column_d2

donor_seq_bed_2_d2.drop(donor_seq_bed_2_d2.loc[:, ['chromStart', 'donorSeqs']], axis=1, inplace=True)

second_column_4_d2 = donor_seq_bed_2_d2.pop('donorCoords')

donor_seq_bed_2_d2.insert(1, 'donorCoords', second_column_4_d2)

subtract_seq_len_d2 = donor_seq_bed_2_d2['donorCoords'] - acceptor_seq_len - 2

donor_seq_bed_2_d2['donorCoords_minus_seq_len'] = subtract_seq_len_d2

add_seq_len_d2 = donor_seq_bed_2_d2['donorCoords'] + acceptor_seq_len

donor_seq_bed_2_d2['donorCoords_plus_seq_len'] = add_seq_len_d2

donor_seq_bed_2_d2 = donor_seq_bed_2_d2.drop('donorCoords', axis=1)

donor_seq_bed_2_d2 = donor_seq_bed_2_d2[['chrom', 'donorCoords_minus_seq_len', 'donorCoords_plus_seq_len', 'name']]

############]if necessary##############
# Add chr to start of every number
# donor_seq_bed_2_d2['chrom'] = 'chr' + donor_seq_bed_2_d2['chrom'].astype(str)
#######################################

g_d2 = donor_seq_bed_2_d2.groupby(['chrom', 'name'])
donor_seq_bed_2_d2.loc[g_d2['name'].transform('size').gt(1),
       'name'] += '_'+g_d2.cumcount().astype(str)

donor_seq_bed_2_d2['name'] = donor_seq_bed_2_d2['name'].astype(str) + '_neg'

filename2 = "/path/to/splice_site_sequences_4.bed"
donor_seq_bed_2_d2.to_csv(filename2, index=False, sep="\t", header=None)

# Check that the file exists before printing the confirmation message
if os.path.isfile(filename2):
    print(f"Saved as {filename2}")
else:
    print(f"Failed to save {filename2}")
