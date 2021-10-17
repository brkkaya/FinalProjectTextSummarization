Final Project of TEDU CMPE Department

Project will have 3 part.

1. Retrieve data from wikipedia via PyMediaWiki API to enhance the quality of trainable data.
   Because, we found a public dataset which created from news(500-1000 words) and it's summary.
   To able to handle longer texts on production, we have to add longer texts.
   Thus, we decided to gather data from wikipedia.
   (In a wikipedia page, first part is the summary. The rest is the detailed text.)
   Therefore, we are able to use wikipedia data on train our state-of-art model.

2. Training a state-of-art deep learning model. We will use PyTorch on training.

3. To able to use trained text summarization model as a product, we need to create a website.
   The website will have two layer. First layer is the website the a client uses.
   The second layer is a python module which makes prediction of the given text file. The first layer and second layer connect within a private IP address to communicate  with each other to get prediction.
   We choose this way because of there is no framework created from PyTorch for JS programs.
   (The website will be created from scratch. The website will be dynamic.)

-Burak Kaya
-Cihan Kutluoğlu
-Muhammet Burak Fazlıoğlu
