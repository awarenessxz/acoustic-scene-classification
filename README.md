## Bonus Section

### Overview

Now that we get the ASC model which can detect scene, we can go further and use it in real life. 
Firstly, what the model tells us about a sound scene is a “label”. For example, we have 10 labels in our current ASC model, and if we use the ASC model to detect a sound scene recorded in park it will very probably return the label as “park”. In this way, this piece of sound can be labeled as “park”. This gives us a possible way to label each time frame of a movie, hence provide a textual description of the movie. In our bonus demo, we implement label classification. Specifically, we choose a local wav file and use the model to predict, the model will return a label index which tells us what kind of scene it is.
Secondly, besides labeling wav files that already be recorded and saved, in more common cases we are interested in the real time sound event. For example, if an old man is going out, we can use this model to predict where he is. It is not like the GPS which only tells us the map location, it will tells us information the real environment and thus provide more information that GPS may not detect. In our demo, we record the environment sound for 10 seconds and the model will tell us where this scene is. 


### Language

We create a website for our bonus section. The front end is written with html, the back end is written with python. 


### Running 

To run the code: `app.py`. 

### Website Designe
We mainly have two functions here. One is choose local audio file and predict. The other is record environment sound and predict. 
Here is the procedure about the website:
Click “Choose File” buttion: Choose a local audio file
Click “Get File” button: Local audio file will be loaded 
Play audio: audio file will play
Click “Guess scene of this audio!”: return label index of local audio file
Write a name of record audio file in “input” and record: start to record environment sound for 10 seconds.
Click “Guess scene of this audio!”: return label index of record audio file
If we choose the local file to predict: the front end will use ajax and “post” audio file path and name to back end. Back end analyze features of the audio file and use CNN model to predict the label of this audio file, after that back end will give back a label index to the front end, front end then alerts the label index.
If we choose to record environment sound: the front end will use ajax and “post” record audio filename to back end. Back end begins to record the sound for 10 seconds and save the record audio in the local project. Then the audio bar will update and load the record file. If we want to know the label we can click “Guess scene of this audio”. We can also integrate the record part and predict part so after recording, the label will immediately alert. While in our demo, we choose to separate these two parts because we want to listen to the record file first. 















