# Question-Answering

This project's aim is to answer geographical questions in Turkish language.
The dataset used for this project will be published later. I will include a link to the dataset when it is published.

The project consists of two tasks
* **Task1:** Finding the paragraph which the answer is in.
* **Task2:** Extracting the answer from the paragraph.

Dataset:

Dataset is prepared from high school geography text books of Turkish National Education Ministry.
It has paragraphs with paragraph ids in the following format:

\<paragraph num> <paragraph1\>

\<paragraph num> <paragraph2\>

...
  
And for the questions:

\<Question id\> \<Question\>
  
\<Answer id\> \<Answer\>
  
Related paragraph: \<paragraph num\>

My method is quiet straightforward. Find the most related N paragraphs using tf_idf model. Then construct sentence representation of every sentence using word2vec models(Any pre-trained word2vec model for Turkish language) and construct a similarity score for all sentences using cosine similarity. Than for every sentence, similarity score is multiplied by the tf_idf similarity score of the containing paragraph. The highest scorer sentence is selected and it's paragraph is answered as the answer to the task1.

After finding the most related sentence, I selected a continuous subsentence and checked if it could be the answer. Checking is done by constructing a sentence representation using word2vec models and checking it's similarity with the question. The highest scorer interval is selected as the answer to the task2.

Although the dataset was not quiet clean, it had some mistakes, it will be refactored and published soon. I had 68 percent accuracy on task1 and 10 percent accuracy on task2. Acquiring the exact same result with the dataset was quiet hard and needs further work because of the nature of the Turkish language. Therefore we used jaccard score as the success metric which we had 31 percent accuracy.

The accuracy can be vastly increased by doing some cleaning in the dataset and employing some pre-processing. When the dataset is refactored and published I am expecting this model to score more than 75 percent accuracy on task1 and more than 50 percent accuracy on task2.

I measured the accuracy of task2 for the successfully found sentences.
