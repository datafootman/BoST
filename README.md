# BoST
We give the ``Bag-of-Senses'' (BoS) assumption that a document is consist of a set of word senses. Based on this assumption, we propose a new topic model, which adapts and captures the various word senses in modeling.

# Usage:
The file tdt2_em_v4_0_100.npy keeps first 100 documents of TDT2 dataset.

The main program is in the "BoST_.py" file.

After the training, the topic distribution, word distribution and the parameter SSWs will be returned and saved under the current folder.

The test program is the file "test_BoST.py".

Run "python test_BoST.py" and test the perplexities for C-LDA and traditional LDA.

# Example:
An example is given in "docs_vec_visualization.py" to show that how to use the generated vectors (or distributions). We provide a visualization example of two trained document vectors by BoST and LDA: "BoST_doc_topic_distributions.npy" and "LDAdoc_topic_distributions.npy". Running "docs_vec_visualization.py" can see the visualization of document vectors (by t-SNE).
