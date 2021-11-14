def batch_read(T, vocab_size,BATCH_SIZE, SEQ_LENGTH ):
    length = T.shape[0]; # length of the text
    batch_chars = int(length / BATCH_SIZE); # number of batches

    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH): # start of the batch
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH)) 
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size)) 
        for batch_idx in range(0, BATCH_SIZE): # batch index
            for i in range(0, SEQ_LENGTH): 
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i] 
                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1
        yield X, Y