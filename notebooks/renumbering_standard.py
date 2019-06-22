def reset_numbering(dataframe,chain):
    
    DFstruct = dataframe.df['ATOM']
    # Store string of resname+insertion
    resninsert = DFstruct.loc[DFstruct["chain_id"] == chain,"residue_number"].map(str) + DFstruct.loc[DFstruct["chain_id"] == chain,"insertion"]
    old_resnames = resninsert.unique().tolist()

    DF_iterchain = DFstruct.loc[DFstruct["chain_id"] == chain,["residue_number","insertion"]]
    first_resnum = DF_iterchain.residue_number.iloc[0]
    # Stoe tuple (newresnum," ")
    tuples_renum = [(i,"") for i in range(first_resnum,DF_iterchain.shape[0]+1)]
    # mapping dict {oldresname:tuple_renum}
    mapping_dict = dict(zip(old_resnames,tuples_renum))
    for i in DF_iterchain.index:
        current_resnumber = str(DF_iterchain.loc[i,"residue_number"]) + DF_iterchain.loc[i,"insertion"]
        if current_resnumber in mapping_dict:
            DF_iterchain.loc[i,["residue_number","insertion"]] = mapping_dict[current_resnumber]
    DFstruct.loc[DFstruct["chain_id"] == chain,["residue_number","insertion"]] = DF_iterchain
    return DFstruct
    