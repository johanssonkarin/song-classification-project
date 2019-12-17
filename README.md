# Music Classification Mini Project

The technical problem was to tell which songs, in a dataset of 200 songs, someone is going to like. In order to do this, a training dataset with 750 songs was provided, each of which is labeled with LIKE or DISLIKE. The dataset consists not of the songs themselves, but of high-level features extracted using [the web-API from Spotify](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/). These high-level features describe characteristics such as the acousticness, danceability, energy, instrumentalness, valence and tempo of each song.

The project covers four classification methods: 
1. Logistic regression
2. Discriminant analysis: LDA, QDA
3. K-nearest neighbor
4. Boosting

Each method was implemented in Python, tuned to perfom well and then evaluated. See details and results in the [report](https://github.com/johanssonkarin/song-classification-project/blob/master/Group%20S181%20Mini%20Project%20Report.pdf). The report also includes a minor reflection task on whether a machine learning engineers have a responsibility to inform and educate clients about the the risk of obtaining (possibly subtle) machine learning biases in the solution.

This is a project from the course in [Statistical Machine Learning](https://www.uu.se/en/admissions/freestanding-courses/course-syllabus/?kpid=41831&lasar=19%2F20&typ=1) at Uppsala University.
