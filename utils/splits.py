def subject_split(metadata, val_subjects, test_subjects):
    test_meta  = metadata[metadata["subject_id"].isin(test_subjects)]
    val_meta   = metadata[metadata["subject_id"].isin(val_subjects)]
    train_meta = metadata[~metadata["subject_id"].isin(
                          val_subjects + test_subjects)]
    return train_meta, val_meta, test_meta