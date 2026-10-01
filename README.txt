🏠 HOUSE PRICE PREDICTOR
========================

HOW TO USE:
-----------
1. Double-click  →  1_SETUP_AND_TRAIN.bat   (do this ONCE — installs packages & trains the model)
2. Double-click  →  2_PREDICT.bat           (terminal app — enter house details and get a price!)
3. Double-click  →  3_RUN_WEB.bat           (web UI — opens http://127.0.0.1:5000 in your browser)

WHAT YOU ENTER:
---------------
  • House size in square feet  (e.g. 1500)
  • Number of bedrooms         (e.g. 3)
  • Number of bathrooms        (e.g. 2)
  • Age of house in years      (e.g. 10)
  • Garage spots               (e.g. 1)
  • Location type              (urban / suburban / rural)

FILES:
------
  generate_dataset.py   →  Creates the training dataset
  train_model.py        →  Trains 4 ML models, saves the best one
  predict.py            →  Terminal prediction app
  app.py                →  Flask web server (frontend)
  templates/index.html  →  Web UI page
  models/best_model.pkl →  Saved trained model
  outputs/              →  Charts and plots from training
