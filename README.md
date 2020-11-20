# Chess Analysis Api Backend (Part of the Project to detect “Turning Points” in Chess matches )

This repository contains the code implementation for the api consumed by the project Chess Analysis Frontend. The former integrated with the latter creates an web app to use an NLP model to detect “Turning Points” in Chess matches, which was a result of the undergraduate thesis on the machine learning area carried out by Douglas Ramos, Pedro Felipe and Rafael Higa under the guidance of Prof. Dr. Glauber de Bona, from University of São Paulo, 2020.

## Project Parts Overview

- Machine Learning (Crawling + Dataset generation + Model training): https://github.com/Rseiji/TCC-2020
- Web Frontend (friendly interface that allows detect the turning points in a chess match, using the trained model): https://github.com/douglasramos/ChessAnalysisFrontend
- Backend Api (Receives a chess match from the frontend, process it using the trained model and returns the result to the frontend): https://github.com/douglasramos/ChessAnalysisApi - This Repo!

## Setup

This is a python 3.7 api and you need to install the following dependencies to make it work:

```
pip install uvicorn
pip install fastapi
```

## How to run

```
uvicorn app.main:app --reload
```

This will expose a web api at http://127.0.0.1:8000/
See http://127.0.0.1:8000/docs for endpoints documentation

## License

Copyright © 2020 Douglas Ramos, Pedro Felipe and Rafael Higa

Distributed under the MIT License, with due credit to the work on which this project was based.
