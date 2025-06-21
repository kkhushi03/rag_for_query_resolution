1. **Unused Weights Warnings**
    ```Some weights of the model checkpoint at facebook/dpr-question_encoder-single-nq-base were not used when initializing DPRQuestionEncoder: ['question_encoder.bert_model.pooler.dense.bias', 'question_encoder.bert_model.pooler.dense.weight']```

    *What it means:*
       * DPR models are based on BERT, but they use a custom pooling strategy
       * The original BERT pooler layers (pooler.dense.bias and pooler.dense.weight) aren't needed for DPR
       * DPR uses its own way to extract the final embedding from BERT's output

    *Why it's expected:*
       * The warning itself says "This IS expected" for models trained on different tasks
       * DPR was specifically trained for dense retrieval, so it modified BERT's architecture
       * These unused weights don't affect functionality at all

2. **Tokenizer Class Warning**
    ```The tokenizer class you load from this checkpoint is 'DPRQuestionEncoderTokenizer'.```
    ```The class this function is called from is 'DPRContextEncoderTokenizer'.```

    *What it means:*
       * DPR has separate tokenizers for questions and contexts
       * The warning occurs because we're loading both tokenizers, and they have slight differences
       * This happens in the _load_dpr_models() function when loading both encoders

    *Why it happens:*
       * Question encoder tokenizer is optimized for short queries
       * Context encoder tokenizer is optimized for longer document passages
       * The warning is just informing you they're different classes