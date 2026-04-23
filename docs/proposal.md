Project #6. Comparing postmortems and practices
The hardest part of this project is creating an unsupervised machine learning model to take post-mortems and distinguish them between “political/business” and “engineering focused”. This line is often blurry, and the goal of the post-mortem is to appease both sides of this aisle. Therefore, another key part of the project is to identify the key features that may sway the post-mortem in a certain direction. Another important aspect is to identify the frequency of certain failures. Finding the “root cause” is simple enough but identifying the underlying goal is not so black and white. 
With any machine learning model, we need a dataset. We need to obtain, possibly scrape, post-mortems from companies and agencies around the world. Important labels will be company, date, title, author if available, and of course the text. A good starting point is https://github.com/danluu/post-mortems?tab=readme-ov-file . Data like stock prices, news headlines, and other external sources could be useful in determining the tone of the post-mortems. 
From there we need to choose the proper model, for this level of complexity something simple like a keyword/document analyzing model could be the answer whereas a small transformer-based model may be able to draw more nuance (politics) from the documents. Anything self-made that is larger would likely be overkill for such a task and not suitable for training with such limited data. 
Possible starting points:
•	TF-IDF + Logistic Regression or SVM 
•	TF-IDF + NMF/LDA
•	BERTopic with all-MiniLM-L6-v2
Using available LLMs may be a good starting point to identify key words. It is important to find an advantage of the system I am creating over simply asking a chat-bot whether a post-mortem is more political or engineering focused. Perhaps, creating a dedicated system allows us to explore the complexity of the issue and could reveal simplicities about intent in writing.
One idea is using an obvious keyword bank of finding the political vs engineering tone then using those labels to identify simpler terms or phrases that people can look for as a politic detector.  

