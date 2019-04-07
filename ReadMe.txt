Steps used to run the program:

1)Install Python 3.x
2)Enter the path of the abstracts folder, gold folder and the value of w as input.
3)Type chmod +x IR4.py to tell Linux that it is an executable program.
4)Type python3 IR4.py to run the program.

About the program:
Considering the entire document as a sequence for creating the graph.

Part 1:
Functions implemented: 
tokenize(): preprocesses the abstracts
preprocess_gold(): preprocesses the gold
create_adj_map(): creates the graph for every document by adding an edge between two adjacent words if within the given window size.

Part 2:
Functions implemented:
PageRank(): run on every document for 10 iterations using alpha = 0.85 and p = 1/n and scores for every node in the document graph are calculated.

Part 3:
Functions implemented:
add_ngrams(): score n-grams or phrases using the sum of scores of individual words that comprise the phrase.
predict_keyphrases(): rank candidate phrases based on their PageRank scores for each document independently.

Part 4:
Functions implemented:
calc_doc_MRR(): calculates the MRR for every doc using the predicted phrases and the gold standard.
calc_global_MRR(): calculate the MRR for k = 1 to 10 for every doc in the abstracts folder.

Output:
MRR values for the WWW collection for w = 6 :

Displaying MRR values for k = 1 to 10...
k = 1	MRR = 0.035338345864661655
k = 2	MRR = 0.0518796992481203
k = 3	MRR = 0.06867167919799497
k = 4	MRR = 0.08126566416040094
k = 5	MRR = 0.09028822055137846
k = 6	MRR = 0.09968671679198005
k = 7	MRR = 0.10688327962764048
k = 8	MRR = 0.11261636233440736
k = 9	MRR = 0.11704409834109084
k = 10	MRR = 0.12005161713808318

