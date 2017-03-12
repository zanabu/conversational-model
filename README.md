# conversational-model

A Brief Description of the Project

This project is an implementation of Google's chat bot, published in the paper A Neural Conversational Model (Oriol Vinyals & Quoc Le). Trained with movie subtitles, the model is capable to respond to questions as intriguing as "What is the meaning of life?". The dataset used is an open domain one (OpenSubtitles). It consists of movie subtitles, where the assumption is that consecutive sentences are uttered by different characters.

The training is done in an end-to-end manner utilizing sequence to sequence framework. This framework works in an encoder-decoder manner. Here, the question is encoded, while the answer is decoded. For the encoding a two-layered Long-Short Term Memory (LSTM) is used. Similarly, for the decoding another two-layered LSTM is used.



