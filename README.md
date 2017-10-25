# utl_machine_learning_deciphering_speech
What was the programmer talking about when he said; "web app", "dog", "super intelligence".  Various word(ngrams) and character analyzers that provide string and substring co-occurrance, sequential patterns, word reversals and character co-occurance? Produces probabilities?

    ```  StackOverflow Python: Machine learning: What was the programmer talking about when he said; "web app", "dog", "super intelligence"                           ```
    ```                                                                                                                                                               ```
    ```  I am way over my head here but this appears to be a very simple example of machine learning                                                                  ```
    ```  to categorize phrases.                                                                                                                                       ```
    ```                                                                                                                                                               ```
    ```    ( Training data )                                                                                                                                          ```
    ```    Independent like variables                                                                                                                                 ```
    ```                                                                                                                                                               ```
    ```        AI                                                                                                                                                     ```
    ```        Artificial Intelligence                                                                                                                                ```
    ```        VR                                                                                                                                                     ```
    ```        Virtual Reality                                                                                                                                        ```
    ```        Mobile application                                                                                                                                     ```
    ```        Desktop softwares                                                                                                                                      ```
    ```                                                                                                                                                               ```
    ```     Dependent Variables                                                                                                                                       ```
    ```                                                                                                                                                               ```
    ```        Artificial Intelligence                                                                                                                                ```
    ```        Artificial Intelligence                                                                                                                                ```
    ```        Virtual Reality                                                                                                                                        ```
    ```        Virtual Reality                                                                                                                                        ```
    ```        Application                                                                                                                                            ```
    ```        Application                                                                                                                                            ```
    ```                                                                                                                                                               ```
    ```     NGRAMS                                                                                                                                                    ```
    ```                                                                                                                                                               ```
    ```        Various word and character analyzers that provide string and substring co-occurrance, sequential patterns,                                             ```
    ```        word reversals and character co-occurance? Produces probabilities?                                                                                     ```
    ```                                                                                                                                                               ```
    ```     BUILD A LOGISTIC MODEL                                                                                                                                    ```
    ```                                                                                                                                                               ```
    ```        Apply the model to new data?                                                                                                                           ```
    ```                                                                                                                                                               ```
    ```                                                                                                                                                               ```
    ```  see                                                                                                                                                          ```
    ```  https://goo.gl/EZpxr7                                                                                                                                        ```
    ```  https://stackoverflow.com/questions/46924600/categories-busineesses-with-text-analytics-in-python                                                            ```
    ```                                                                                                                                                               ```
    ```  Jarad profile                                                                                                                                                ```
    ```  https://stackoverflow.com/users/1577947/jarad                                                                                                                ```
    ```                                                                                                                                                               ```
    ```  also see                                                                                                                                                     ```
    ```                                                                                                                                                               ```
    ```                                                                                                                                                               ```
    ```  HAVE                                                                                                                                                         ```
    ```  ====                                                                                                                                                         ```
    ```     "web application", "web app", "dog", "super intelligence"                                                                                                 ```
    ```                                                                                                                                                               ```
    ```     and a model built using                                                                                                                                   ```
    ```                                                                                                                                                               ```
    ```     Independent like variables                                                                                                                                ```
    ```                                                                                                                                                               ```
    ```        AI                                                                                                                                                     ```
    ```        Artificial Intelligence                                                                                                                                ```
    ```        VR                                                                                                                                                     ```
    ```        Virtual Reality                                                                                                                                        ```
    ```        Mobile application                                                                                                                                     ```
    ```        Desktop softwares                                                                                                                                      ```
    ```                                                                                                                                                               ```
    ```      Dependent Variables                                                                                                                                      ```
    ```                                                                                                                                                               ```
    ```         Artificial Intelligence                                                                                                                               ```
    ```         Artificial Intelligence                                                                                                                               ```
    ```         Virtual Reality                                                                                                                                       ```
    ```         Virtual Reality                                                                                                                                       ```
    ```         Application                                                                                                                                           ```
    ```         Application                                                                                                                                           ```
    ```                                                                                                                                                               ```
    ```                                                                                                                                                               ```
    ```  WANT                                                                                                                                                         ```
    ```  ===                                                                                                                                                          ```
    ```      Programmers Speech             Deciphered                                                                                                                ```
    ```      ==================             ===========                                                                                                               ```
    ```                                                                                                                                                               ```
    ```      web application       maps to  Application                                                                                                               ```
    ```      web app               maps to  Application                                                                                                               ```
    ```      dog                   maps to  Virtual Reality                                                                                                           ```
    ```      super intelligence    maps to  Artificial Intelligence                                                                                                   ```
    ```                                                                                                                                                               ```
    ```  *                _               _       _                                                                                                                   ```
    ```   _ __ ___   __ _| | _____     __| | __ _| |_ __ _                                                                                                            ```
    ```  | '_ ` _ \ / _` | |/ / _ \   / _` |/ _` | __/ _` |                                                                                                           ```
    ```  | | | | | | (_| |   <  __/  | (_| | (_| | || (_| |                                                                                                           ```
    ```  |_| |_| |_|\__,_|_|\_\___|   \__,_|\__,_|\__\__,_|                                                                                                           ```
    ```                                                                                                                                                               ```
    ```  ;                                                                                                                                                            ```
    ```                                                                                                                                                               ```
    ```    All data is imbedded in python script                                                                                                                      ```
    ```  *          _       _   _                                                                                                                                     ```
    ```   ___  ___ | |_   _| |_(_) ___  _ __                                                                                                                          ```
    ```  / __|/ _ \| | | | | __| |/ _ \| '_ \                                                                                                                         ```
    ```  \__ \ (_) | | |_| | |_| | (_) | | | |                                                                                                                        ```
    ```  |___/\___/|_|\__,_|\__|_|\___/|_| |_|                                                                                                                        ```
    ```                                                                                                                                                               ```
    ```  ;                                                                                                                                                            ```
    ```                                                                                                                                                               ```
    ```  %utl_submit_py64('                                                                                                                                           ```
    ```  import numpy as np;                                                                                                                                          ```
    ```  from sklearn.linear_model import LogisticRegression;                                                                                                         ```
    ```  from sklearn.feature_extraction.text import CountVectorizer;                                                                                                 ```
    ```  from sklearn.pipeline import Pipeline, FeatureUnion;                                                                                                         ```
    ```  X = np.array([["AI"],;                                                                                                                                       ```
    ```        ["Artificial Intelligence"],;                                                                                                                          ```
    ```        ["VR"],;                                                                                                                                               ```
    ```        ["Virtual Reality"],;                                                                                                                                  ```
    ```        ["Mobile application"],;                                                                                                                               ```
    ```        ["Desktop softwares"]]);                                                                                                                               ```
    ```  y = np.array(["Artificial Intelligence", "Artificial Intelligence",;                                                                                         ```
    ```  .      "Virtual Reality", "Virtual Reality", "Application", "Application"]);                                                                                 ```
    ```  pipeline = Pipeline(steps=[;                                                                                                                                 ```
    ```  .   ("union", FeatureUnion([;                                                                                                                                ```
    ```  .       ("word_vec", CountVectorizer(binary=True, analyzer="word", ngram_range=(1,2))),;                                                                     ```
    ```  .       ("char_vec", CountVectorizer(analyzer="char", ngram_range=(2,5)));                                                                                   ```
    ```  .       ])),;                                                                                                                                                ```
    ```  .   ("lreg", LogisticRegression());                                                                                                                          ```
    ```  .   ]);                                                                                                                                                      ```
    ```  pipeline.fit(X.ravel(), y);                                                                                                                                  ```
    ```  print(pipeline.predict(["web application", "web app", "dog", "super intelligence"]));                                                                        ```
    ```  ');                                                                                                                                                          ```
    ```                                                                                                                                                               ```
    ```                                                                                                                                                               ```
    ```  ['Application' 'Application' 'Virtual Reality' 'Artificial Intelligence']                                                                                    ```
    ```                                                                                                                                                               ```
    ```                                                                                                                                                               ```
    ```  1211  %utl_submit_py64('                                                                                                                                     ```
    ```  MLOGIC(UTL_SUBMIT_PY64):  Beginning execution.                                                                                                               ```
    ```  MLOGIC(UTL_SUBMIT_PY64):  This macro was compiled from the autocall file c:\oto\utl_submit_py64.sas                                                          ```
    ```  1212  import numpy as np;                                                                                                                                    ```
    ```  1213  from sklearn.linear_model import LogisticRegression;                                                                                                   ```
    ```  1214  from sklearn.feature_extraction.text import CountVectorizer;                                                                                           ```
    ```  1215  from sklearn.pipeline import Pipeline, FeatureUnion;                                                                                                   ```
    ```  1216  X = np.array([["AI"],;                                                                                                                                 ```
    ```  1217        ["Artificial Intelligence"],;                                                                                                                    ```
    ```  1218        ["VR"],;                                                                                                                                         ```
    ```  1219        ["Virtual Reality"],;                                                                                                                            ```
    ```  1220        ["Mobile application"],;                                                                                                                         ```
    ```  1221        ["Desktop softwares"]]);                                                                                                                         ```
    ```  1222  y = np.array(["Artificial Intelligence", "Artificial Intelligence",;                                                                                   ```
    ```  1223  .      "Virtual Reality", "Virtual Reality", "Application", "Application"]);                                                                           ```
    ```  1224  pipeline = Pipeline(steps=[;                                                                                                                           ```
    ```  1225  .   ("union", FeatureUnion([;                                                                                                                          ```
    ```  1226  .       ("word_vec", CountVectorizer(binary=True, analyzer="word", ngram_range=(1,2))),;                                                               ```
    ```  1227  .       ("char_vec", CountVectorizer(analyzer="char", ngram_range=(2,5)));                                                                             ```
    ```  1228  .       ])),;                                                                                                                                          ```
    ```  1229  .   ("lreg", LogisticRegression());                                                                                                                    ```
    ```  1230  .   ]);                                                                                                                                                ```
    ```  1231  pipeline.fit(X.ravel(), y);                                                                                                                            ```
    ```  1232  print(pipeline.predict(["web application", "web app", "dog", "super intelligence"]));                                                                  ```
    ```  1233  ');                                                                                                                                                    ```
    ```  MLOGIC(UTL_SUBMIT_PY64):  Parameter PGM has value 'import numpy as np;from sklearn.linear_model import                                                       ```
    ```  LogisticRegression;from sklearn.feature_extraction.text import                                                                                               ```
    ```        CountVectorizer;from sklearn.pipeline import Pipeline, FeatureUnion;X = np.array([["AI"],;                                                             ```
    ```  ["Artificial Intelligence"],;      ["VR"],;      ["Virtual                                                                                                   ```
    ```        Reality"],;      ["Mobile application"],;      ["Desktop softwares"]]);y = np.array(["Artificial                                                       ```
    ```  Intelligence", "Artificial Intelligence",;.      "Virtual Reality",                                                                                          ```
    ```        "Virtual Reality", "Application", "Application"]);pipeline = Pipeline(steps=[;.   ("union",                                                            ```
    ```  FeatureUnion([;.       ("word_vec", CountVectorizer(binary=True,                                                                                             ```
    ```        analyzer="word", ngram_range=(1,2))),;.       ("char_vec", CountVectorizer(analyzer="char",                                                            ```
    ```  ngram_range=(2,5)));.       ])),;.   ("lreg", LogisticRegression());.                                                                                        ```
    ```        ]);pipeline.fit(X.ravel(), y);print(pipeline.predict(["web application", "web app", "dog",                                                             ```
    ```  "super intelligence"]));'                                                                                                                                    ```
    ```                                                                                                                                                               ```
    ```  SYMBOLGEN:  Macro variable _LOC resolves to e:\saswork\wrk\_TD4900_BEAST_\py_pgm.py                                                                          ```
    ```  e:\saswork\wrk\_TD4900_BEAST_\py_pgm.py                                                                                                                      ```
    ```  SYMBOLGEN:  Macro variable _LOC resolves to e:\saswork\wrk\_TD4900_BEAST_\py_pgm.py                                                                          ```
    ```  MPRINT(UTL_SUBMIT_PY64):   filename rut pipe "C:\Python_27_64bit/python.exe e:\saswork\wrk\_TD4900_BEAST_\py_pgm.py";                                        ```
    ```  MPRINT(UTL_SUBMIT_PY64):   data _null_;                                                                                                                      ```
    ```  MPRINT(UTL_SUBMIT_PY64):   file print;                                                                                                                       ```
    ```  MPRINT(UTL_SUBMIT_PY64):   infile rut;                                                                                                                       ```
    ```  MPRINT(UTL_SUBMIT_PY64):   input;                                                                                                                            ```
    ```  MPRINT(UTL_SUBMIT_PY64):   put _infile_;                                                                                                                     ```
    ```  MPRINT(UTL_SUBMIT_PY64):   run;                                                                                                                              ```
    ```                                                                                                                                                               ```
    ```                                                                                                                                                               ```
    ```                                                                                                                                                               ```

